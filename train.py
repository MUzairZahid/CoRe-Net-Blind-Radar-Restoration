import argparse
import os
import torch
import torch.nn as nn
from utils import *
from torch.utils.tensorboard import SummaryWriter


def train_coreNet(dataloaders, A, M, args):

    # Check if a saved model exists and load it if continuing training
    if args.continue_training and os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=args.device)
        A.load_state_dict(checkpoint['A_state_dict'])
        M.load_state_dict(checkpoint['M_state_dict'])
        print(f"Loaded model weights from {args.model_path}")

    # Initialize SummaryWriter for TensorBoard
    writer = SummaryWriter(log_dir = os.path.join(args.out_folder, 'runs'))

    betas=(0.5, 0.999)
    A_params = list(A.parameters())
    M_params = list(M.parameters())
    # Initialize optimizers for Generator and Discriminator
    optimizer_A = torch.optim.Adam(A_params, lr = args.lr_A, betas=betas)
    optimizer_M = torch.optim.Adam(M_params, lr= args.lr_M, betas=betas)

    # Initialize learning rate schedulers if specified
    if args.scheduler == "cosine":
        scheduler_A = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_A, T_max=args.T_max, eta_min=0.00005)
        scheduler_M = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_M, T_max=args.T_max, eta_min=0.00005)
    elif args.scheduler == "cosineWarm":
        scheduler_A = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_A, T_0=25, T_mult=2, eta_min=0)
        scheduler_M = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_M, T_0=25, T_mult=2, eta_min=0)
    else:
        scheduler_A = None
        scheduler_M = None

    if args.rec_loss == "psnr":
        criterion_rec = PSNRLoss()
    elif args.rec_loss == "l1":
        criterion_rec = nn.L1Loss()
    elif args.rec_loss == "mse":
        criterion_rec = nn.MSELoss()

    if args.psnr_pred_loss == "mse":
        criterion_psnr_pred = nn.MSELoss()
    elif args.psnr_pred_loss == "l1":
        criterion_psnr_pred = nn.L1Loss()

    best_validation_snr = float('-inf')  # Initialize the best validation SNR
    # early_stopping = EarlyStopping(patience=200)


    for epoch in range(args.epochs):
        # Training phase
        train_snr, lr = run_epoch(A, M, dataloaders, optimizer_A, optimizer_M, scheduler_A, scheduler_M, criterion_rec, criterion_psnr_pred, args, writer, epoch, phase='train')

        
        if epoch % args.val_interval == 0:

            # Validation phase
            val_snr, _ = run_epoch(A, M, dataloaders, optimizer_A, optimizer_M, scheduler_A, scheduler_M, criterion_rec, criterion_psnr_pred, args, writer, epoch, phase='validation')

            print('-'*100)
            print(f'Epoch [{epoch+1}/{args.epochs}], Train SNR: {train_snr:.2f}, Validation SNR: {val_snr:.2f},  lr: {lr:.6f}')

            with open(os.path.join(args.out_folder, f'snr_log.txt'), 'a') as f:
                    f.write(f'Epoch [{epoch+1}/{args.epochs}], Train SNR: {train_snr:.2f}, Validation SNR: {val_snr:.2f},  lr: {lr:.6f}\n')


            if val_snr > best_validation_snr:
                print(f'Validation SNR improved from: {best_validation_snr:.2f}, to {val_snr:.2f}')
                best_validation_snr = val_snr
                save_best_model(A, M, args.model_path)

    
    # Close the SummaryWriter
    writer.close()


def run_epoch(
    A, M, dataloaders,
    optimizer_A, optimizer_M,
    scheduler_A, scheduler_M,
    criterion_rec,           # e.g., L1/MSE for time/freq branches
    criterion_master,        # e.g., MSE for MR fidelity/targets
    args, writer, epoch, phase='train'
):
    """
    Original CoRe-Net training loop (no classifier):
      - Apprentice Regressor (AR) is guided cooperatively by Master Regressor (MR).
      - Two-stage update per batch:
          1) AR step: MR is FROZEN; gradients flow through M(r, ŝ) to AR only.
          2) MR step: AR is FROZEN; MR regresses normalized PSNR targets.

    Returns:
      average_snr (validation only; 0.0 on train), lr
    """

    # ---------- helpers ----------
    def freeze(net, flag=True):
        for p in net.parameters():
            p.requires_grad = not flag

    # ---------- trackers (epoch aggregates) ----------
    # AR losses
    loss_A_sum            = 0.0
    loss_A_fid_sum        = 0.0
    loss_A_time_sum       = 0.0
    loss_A_freq_sum       = 0.0

    # MR losses
    loss_M_sum            = 0.0
    loss_M_real_sum       = 0.0
    loss_M_fake_sum       = 0.0

    # MR outputs (for monitoring calibration)
    mean_out_M_real_sum   = 0.0
    mean_out_M_fake_sum   = 0.0

    # Target means (for sanity checks)
    mean_lab_real_sum     = 0.0
    mean_lab_fake_sum     = 0.0

    # Restoration metric (kept off the hot path; val only)
    total_snr = 0.0
    num_snrs  = 0

    total_samples = 0

    # ---------- modes ----------
    if phase == 'train':
        A.train(); M.train()
    else:
        A.eval();  M.eval()


    # Use grad only when training
    use_grad = (phase == 'train')
    with torch.set_grad_enabled(use_grad):
        for (clean_signal_real, clean_min, clean_max), \
            (distorted_signal, distorted_min, distorted_max), \
            labels_unused, snr_distorted_unused, distortions_unused in dataloaders[phase]:

            B = clean_signal_real.size(0)
            total_samples += B

            # ---------- move to device ----------
            clean_signal_real = clean_signal_real.to(args.device, non_blocking=True)
            distorted_signal  = distorted_signal.to(args.device,  non_blocking=True)
            clean_min         = clean_min.to(args.device,         non_blocking=True)
            clean_max         = clean_max.to(args.device,         non_blocking=True)

            # ============================================================
            #  (1) AR STEP — MR FROZEN; grads flow through M(r, ŝ) to AR
            # ============================================================
            freeze(M, True)  # <-- freeze MR params; DO NOT use no_grad (we need the graph for ŝ)

            # Forward AR: r -> ŝ
            clean_signal_fake = A(distorted_signal)

            # MR fidelity term: make M(r, ŝ) ~ 1 (high quality)
            out_M_fake_for_AR = M(clean_signal_fake, distorted_signal)  # grads flow to ŝ (and thus A), not into M
            real_labels = torch.ones(B, 1, device=args.device)          # target "1" = max normalized PSNR
            L_fid = criterion_master(out_M_fake_for_AR, real_labels)

            # Time-domain reconstruction (e.g., L1 / MSE)
            L_time = criterion_rec(clean_signal_fake, clean_signal_real)


            # Frequency-domain reconstruction only if needed
            if args.Phi > 1e-12:
                # Frequency-domain reconstruction (optional, e.g., L1 on log-mag spectrograms)
                spec_clean = calculate_spectrogram_torch(clean_signal_real)  # Spectrogram of the clean (real) signal
                spec_fake = calculate_spectrogram_torch(clean_signal_fake)  # Spectrogram of the restored (fake) signal
                L_freq = criterion_rec(spec_fake, spec_clean)
            else:
                L_freq = clean_signal_fake.new_tensor(0.0)

            # Total AR objective
            loss_A = args.Eps * L_fid + args.Beta * L_time + args.Phi * L_freq


            if phase == 'train':
                optimizer_A.zero_grad(set_to_none=True)
                loss_A.backward()       # <-- no retain_graph
                # Optional: clip AR grads if unstable
                # torch.nn.utils.clip_grad_norm_(A.parameters(), 1.0)
                optimizer_A.step()

            # Log AR aggregates
            loss_A_sum      += loss_A.item()      * B
            loss_A_fid_sum  += (args.Eps  * L_fid).item()  * B
            loss_A_time_sum += (args.Beta * L_time).item() * B
            loss_A_freq_sum += (args.Phi  * L_freq).item() * B

            # ============================================================
            #  (2) MR STEP — AR FROZEN; train MR to regress PSNR targets
            # ============================================================
            freeze(M, False)  # unfreeze MR for its update

            # Build normalized PSNR targets (detached to avoid leaking AR grads)
            with torch.no_grad():
                # α_fake = PSNR(ŝ, s) / targetpsnr; α_dist = PSNR(r, s) / targetpsnr
                # Ensure calculate_psnr uses a consistent MAX across your pipeline
                alpha_clean_fake = calculate_psnr(clean_signal_fake, clean_signal_real).unsqueeze(1) / args.targetpsnr
                #alpha_clean_fake = alpha_clean_fake.clamp(0.0, 1.0)
                fake_labels      = alpha_clean_fake


            # MR supervision on clean (r, s) -> 1 and restored (r, ŝ.detach()) -> α_fake
            out_M_real = M(clean_signal_real,              distorted_signal)
            out_M_fake = M(clean_signal_fake.detach(),     distorted_signal)

            L_M_real = criterion_master(out_M_real, real_labels)
            L_M_fake = criterion_master(out_M_fake, fake_labels)


            loss_M     = (L_M_real + L_M_fake) / 2.0

            if phase == 'train':
                optimizer_M.zero_grad(set_to_none=True)
                loss_M.backward()
                optimizer_M.step()

            # Log MR aggregates 
            scale = float(getattr(args, "targetpsnr", 1.0))
            loss_M_sum      += scale * loss_M.item()      * B
            loss_M_real_sum += scale * L_M_real.item()    * B
            loss_M_fake_sum += scale * L_M_fake.item()    * B

            mean_out_M_real_sum += scale * out_M_real.mean().item() * B
            mean_lab_real_sum   += scale * real_labels.mean().item() * B
            mean_out_M_fake_sum += scale * out_M_fake.mean().item() * B
            mean_lab_fake_sum   += scale * fake_labels.mean().item() * B


            # ---------------- metrics ----------------
            with torch.no_grad():
                restored = denormalize_signal(clean_signal_fake.detach(), clean_min, clean_max)
                clean    = denormalize_signal(clean_signal_real.detach(), clean_min, clean_max)
                batch_snr = calculate_snr(restored, clean)
                total_snr += batch_snr.sum().item()
                num_snrs  += B

    # ---------- schedulers (once per epoch) ----------
    if phase == 'train':
        if scheduler_A is not None: scheduler_A.step()
        if scheduler_M is not None: scheduler_M.step()

    # ---------- learning rate for logging ----------
    lr = optimizer_A.param_groups[0]['lr'] if phase == 'train' else None

    # ---------- averages ----------
    denom = max(1, total_samples)
    avg_LA      = loss_A_sum      / denom
    avg_LA_fid  = loss_A_fid_sum  / denom
    avg_LA_time = loss_A_time_sum / denom
    avg_LA_freq = loss_A_freq_sum / denom

    avg_LM      = loss_M_sum      / denom
    avg_LM_real = loss_M_real_sum / denom
    avg_LM_fake = loss_M_fake_sum / denom

    avg_out_real = mean_out_M_real_sum / denom
    avg_lab_real = mean_lab_real_sum   / denom
    avg_out_fake = mean_out_M_fake_sum / denom
    avg_lab_fake = mean_lab_fake_sum   / denom

    average_snr = (total_snr / num_snrs) if num_snrs > 0 else 0.0

    # ---------- TensorBoard logging ----------
    write_losses_to_tensorboard(
        # AR block
        avg_LA, avg_LA_fid, avg_LA_time, avg_LA_freq,
        # MR block
        avg_LM, avg_LM_real, avg_LM_fake,
        # MR outputs vs targets
        avg_out_real, avg_lab_real, avg_out_fake, avg_lab_fake,
        writer, epoch, phase, lr
    )

    return average_snr, lr
