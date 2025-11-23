import os
import argparse
import torch
import torch.nn.functional as F
import torchaudio.transforms as AT
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import set_seed
from accelerate import Accelerator
import matplotlib.pyplot as plt

from model import Tacotron2, Tacotron2Config
from dataset import TTSDataset, TTSCollator, BatchSampler, denormalize
from tokenizer import Tokenizer

def parse_args():

    parser = argparse.ArgumentParser()

    ### SETUP CONFIG ###
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--working_directory", type=str, required=True)
    parser.add_argument("--save_audio_gen", type=str, required=True)
    parser.add_argument("--path_to_train_manifest", type=str, required=True)
    parser.add_argument("--path_to_val_manifest", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    ### TRAINING CONFIG ###
    parser.add_argument("--training_epochs", type=int, default=500)
    parser.add_argument("--console_out_iters", type=int, default=5)
    parser.add_argument("--wandb_log_iters", type=int, default=5)
    parser.add_argument("--checkpoint_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--start_decay_epochs", type=int, default=None)

    ### MODEL CONFIG ###
    parser.add_argument("--character_embed_dim", type=int, default=512)
    parser.add_argument("--encoder_kernel_size", type=int, default=5)
    parser.add_argument("--encoder_n_convolutions", type=int, default=3)
    parser.add_argument("--encoder_embed_dim", type=int, default=512)
    parser.add_argument("--encoder_dropout_p", type=float, default=0.5)
    parser.add_argument("--decoder_rnn_embed_dim", type=int, default=1024)
    parser.add_argument("--decoder_dropout_p", type=float, default=0.1)
    parser.add_argument("--decoder_prenet_dim", type=int, default=256)
    parser.add_argument("--decoder_prenet_depth", type=int, default=2)
    parser.add_argument("--decoder_prenet_dropout_p", type=float, default=0.5)
    parser.add_argument("--decoder_postnet_num_convs", type=int, default=5)
    parser.add_argument("--decoder_postnet_n_filters", type=int, default=512)
    parser.add_argument("--decoder_postnet_kernel_size", type=int, default=5)
    parser.add_argument("--decoder_postnet_dropout_p", type=float, default=0.5)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--attention_dropout_p", type=float, default=0.1)
    parser.add_argument("--attention_location_n_filters", type=int, default=32)
    parser.add_argument("--attention_location_kernel_size", type=int, default=31)
    
    ### DATASET CONFIG ###
    parser.add_argument("--sampling_rate", type=int, default=22050)
    parser.add_argument("--num_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=256)
    parser.add_argument("--min_db", type=float, default=-100.0)
    parser.add_argument("--max_scaled_abs", type=float, default=1.0)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=8000)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction)

    return parser.parse_args()

### Parser Arguments ###
args = parse_args()

### Set Seed ###
if args.seed is not None:
    set_seed(args.seed)

### Init Accelerator ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                          log_with="wandb" if args.log_wandb else None)

if args.log_wandb:
    accelerator.init_trackers(
        project_name=args.experiment_name, init_kwargs={"wandb":{"name":args.run_name}}
    )

accelerator.print(args)

### Create Paths for Gen Saves ###
if accelerator.is_main_process:
    os.makedirs(args.save_audio_gen, exist_ok=True)

### Load Tokenizer ###
tokenizer = Tokenizer()

### Load Model ###
config = Tacotron2Config(
    num_mels=args.num_mels,
    num_chars=tokenizer.vocab_size, 
    character_embed_dim=args.character_embed_dim,
    pad_token_id=tokenizer.pad_token_id,
    encoder_kernel_size=args.encoder_kernel_size,
    encoder_n_convolutions=args.encoder_n_convolutions,
    encoder_embed_dim=args.encoder_embed_dim,
    encoder_dropout_p=args.encoder_dropout_p,
    decoder_embed_dim=args.decoder_rnn_embed_dim,
    decoder_dropout_p=args.decoder_dropout_p,
    decoder_prenet_dim=args.decoder_prenet_dim,
    decoder_prenet_depth=args.decoder_prenet_depth,
    decoder_prenet_dropout_p=args.decoder_prenet_dropout_p,
    decoder_postnet_num_convs=args.decoder_postnet_num_convs,
    decoder_postnet_n_filters=args.decoder_postnet_n_filters,
    decoder_postnet_kernel_size=args.decoder_postnet_kernel_size,
    decoder_postnet_dropout_p=args.decoder_postnet_dropout_p,
    attention_dim=args.attention_dim,
    attention_dropout_p=args.attention_dropout_p,
    attention_location_n_filters=args.attention_location_n_filters,
    attention_location_kernel_size=args.attention_location_kernel_size
)

model = Tacotron2(config) 
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
accelerator.print(f"Total Trainable Parameters: {total_trainable_params}")

### Load Optimizer ###
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=args.learning_rate, 
                             weight_decay=args.weight_decay, 
                             eps=args.adam_eps)

### Load Dataset ###
trainset = TTSDataset(args.path_to_train_manifest, 
                      sample_rate=args.sampling_rate, 
                      n_fft=args.n_fft, 
                      window_size=args.window_size, 
                      hop_size=args.hop_size, 
                      fmin=args.fmin, 
                      fmax=args.fmax, 
                      num_mels=args.num_mels, 
                      min_db=args.min_db, 
                      max_scaled_abs=args.max_scaled_abs)

testset = TTSDataset(args.path_to_val_manifest, 
                      sample_rate=args.sampling_rate, 
                      n_fft=args.n_fft, 
                      window_size=args.window_size, 
                      hop_size=args.hop_size, 
                      fmin=args.fmin, 
                      fmax=args.fmax, 
                      num_mels=args.num_mels, 
                      min_db=args.min_db, 
                      max_scaled_abs=args.max_scaled_abs)

collator = TTSCollator()
train_sampler = BatchSampler(trainset, 
                             batch_size=args.batch_size, 
                             drop_last=accelerator.num_processes > 1)

trainloader = DataLoader(trainset, 
                         batch_sampler=train_sampler, 
                         num_workers=args.num_workers,
                         collate_fn=collator)

testloader = DataLoader(testset, 
                        batch_size=args.batch_size, 
                        num_workers=args.num_workers,
                        collate_fn=collator)

### Prepare Everything ###
model, optimizer, trainloader, testloader = accelerator.prepare(
    model, optimizer, trainloader, testloader
)

### Create Scheduler ###
using_scheduler = False
if args.start_decay_epochs is not None:
    accelerator.print("Using LR Scheduler!!")
    using_scheduler = True
    init_lr = args.learning_rate
    min_lr = args.min_learning_rate
    decay_epochs = args.training_epochs - args.start_decay_epochs
    decay_gamma = (min_lr / init_lr) ** (1 / decay_epochs)

    def lr_lambda(epoch):
        if epoch < args.start_decay_epochs:
            return 1.0
        else:
            return decay_gamma ** (epoch - args.start_decay_epochs)

### Load Checkpoint ###
if args.resume_from_checkpoint is not None:

    ### Grab path to checkpoint ###
    path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)

    with accelerator.main_process_first():
        accelerator.load_state(path_to_checkpoint)
    
    ### Start completed steps from checkpoint index ###
    completed_epochs = int(args.resume_from_checkpoint.split("_")[-1]) + 1
    completed_steps = completed_epochs * len(trainloader)
    accelerator.print(f"Resuming from Epoch: {completed_epochs}")

    if using_scheduler:
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=completed_epochs-1)

else:
    completed_epochs = 0
    completed_steps = 0

    if using_scheduler:
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

### Train Model ###
for epoch in range(completed_epochs, args.training_epochs):
    
    accelerator.print(f"Epoch: {epoch}")

    model.train()
    for texts, text_lens, mels, stops, encoder_mask, decoder_mask in trainloader:
      
        texts = texts.to(accelerator.device)
        mels = mels.to(accelerator.device)
        stops = stops.to(accelerator.device)
        encoder_mask = encoder_mask.to(accelerator.device)
        decoder_mask = decoder_mask.to(accelerator.device)
        
        ### Generate Mel Spectrogram from Text ###
        mels_out, mels_postnet_out, stop_preds, _ = model(
            texts, text_lens.to("cpu"), mels, encoder_mask, decoder_mask
        )

        ### Compute Loss ###
        mel_loss = F.mse_loss(mels_out, mels)
        refined_mel_loss = F.mse_loss(mels_postnet_out, mels)
        stop_loss = F.binary_cross_entropy_with_logits(stop_preds.reshape(-1,1), stops.reshape(-1,1))

        loss = mel_loss + refined_mel_loss + stop_loss

        ### Update Model ###
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        ### Grab Metrics from all GPUs for Logging ###
        loss = torch.mean(accelerator.gather_for_metrics(loss)).item()
        mel_loss = torch.mean(accelerator.gather_for_metrics(mel_loss)).item()
        refined_mel_loss = torch.mean(accelerator.gather_for_metrics(refined_mel_loss)).item()
        stop_loss = torch.mean(accelerator.gather_for_metrics(stop_loss)).item()

        if completed_steps % args.console_out_iters == 0:
            accelerator.print("Completed Steps {}/{} | Loss {:.4f} | Mel Loss {:.4f} | RMel Loss {:.4f} | Stop Loss {:.4f}".format(
                completed_steps, 
                args.training_epochs * len(trainloader), 
                loss, 
                mel_loss, 
                refined_mel_loss, 
                stop_loss
            ))
       
        if completed_steps % args.wandb_log_iters == 0:
            
            if args.log_wandb:
                accelerator.log(
                    {
                        "mel_loss": mel_loss, 
                        "refined_mel_loss": refined_mel_loss, 
                        "stop_loss": stop_loss,
                        "total_loss": loss
                    }, 
                    step=completed_steps
                )
     
        completed_steps +=1 

    accelerator.wait_for_everyone()

    ### Evaluate Model ###
    model.eval()
    accelerator.print("--VALIDATION--")
    val_mel_loss, val_rmel_loss, val_stop_loss, num_losses = 0, 0, 0, 0
    save_first = True
    for texts, text_lens, mels, stops, encoder_mask, decoder_mask in testloader:
        
        texts = texts.to(accelerator.device)
        mels = mels.to(accelerator.device)
        stops = stops.to(accelerator.device)
        encoder_mask = encoder_mask.to(accelerator.device)
        decoder_mask = decoder_mask.to(accelerator.device)

        ### Generate Mel Spectrogram from Text ###
        with torch.no_grad():
            mels_out, mels_postnet_out, stop_preds, attention_weights = model(
                texts, text_lens.to("cpu"), mels, encoder_mask, decoder_mask
            )

        ### Compute Loss ###
        mel_loss = F.mse_loss(mels_out, mels)
        refined_mel_loss = F.mse_loss(mels_postnet_out, mels)
        stop_loss = F.binary_cross_entropy_with_logits(stop_preds.reshape(-1,1), stops.reshape(-1,1))

        val_mel_loss += mel_loss
        val_rmel_loss += refined_mel_loss
        val_stop_loss += stop_loss
        num_losses += 1

        if accelerator.is_main_process:
            if save_first:

                # Extract tensors
                true_mel = denormalize(mels[0].T.to("cpu"))
                pred_mel = denormalize(mels_postnet_out[0].T.to("cpu"))
                attention = attention_weights[0].T.to("cpu")

                # Make subplots (3 rows, 1 column)
                fig, axes = plt.subplots(3, 1, figsize=(8, 12))
                
                # True Mel
                im0 = axes[0].imshow(true_mel, aspect='auto', origin='lower', interpolation='none')
                axes[0].set_title("True Mel")
                axes[0].set_ylabel("Mel bins")
                fig.colorbar(im0, ax=axes[0])

                # Predicted Mel
                im1 = axes[1].imshow(pred_mel, aspect='auto', origin='lower', interpolation='none')
                axes[1].set_title("Predicted Mel")
                axes[1].set_ylabel("Mel bins")
                fig.colorbar(im1, ax=axes[1])

                # Attention
                im2 = axes[2].imshow(attention, aspect='auto', origin='lower', interpolation='none')
                axes[2].set_title("Alignment")
                axes[2].set_ylabel("Character Index")
                axes[2].set_xlabel("Decoder Mel Timesteps")
                fig.colorbar(im2, ax=axes[2])

                # Adjust layout
                plt.tight_layout()

                # Save combined figure
                plt.savefig(os.path.join(args.save_audio_gen, f"epoch_{epoch}_result.png"))

                plt.close()
        
        save_first = False
    
    val_mel_loss = torch.mean(accelerator.gather_for_metrics(val_mel_loss)).item() / num_losses
    val_rmel_loss = torch.mean(accelerator.gather_for_metrics(val_rmel_loss)).item() / num_losses
    val_stop_loss = torch.mean(accelerator.gather_for_metrics(val_stop_loss)).item() / num_losses
    val_loss = val_mel_loss + val_rmel_loss + val_stop_loss
    
    accelerator.print("Loss {:.4f} | Mel Loss {:.4f} | RMel Loss {:.4f} | Stop Loss {:.4f}".format(
                val_loss, 
                val_mel_loss, 
                val_rmel_loss, 
                val_stop_loss
            ))
    
    
    if args.log_wandb:
        
        accelerator.log(
                    {
                        "val_mel_loss": val_mel_loss, 
                        "val_refined_mel_loss": val_rmel_loss, 
                        "val_stop_loss": val_stop_loss,
                        "val_total_loss": val_loss
                    }, 
                    step=completed_steps
        )
    
    if completed_epochs % args.checkpoint_epochs == 0:
        accelerator.print("Saving Checkpoint!")
        path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{completed_epochs}")
        accelerator.save_state(output_dir=path_to_checkpoint, safe_serialization=False)
    
    completed_epochs += 1

    if using_scheduler:
        scheduler.step(epoch=completed_epochs)
        accelerator.print(f"Learning Rate: {scheduler.get_last_lr()[0]}")

accelerator.save_state(os.path.join(path_to_experiment, "final_checkpoint"), safe_serialization=False)

accelerator.end_training()