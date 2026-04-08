import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, default_data_collator
from torch.optim import AdamW
import jiwer
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import math
import argparse
from sklearn.model_selection import train_test_split
from transformers import logging
logging.set_verbosity_error()


EPOCH = 200
PATIENCE = 20
BATCH_SIZE = 24


def load_data(gt_path, image_dir=None, is_unified=False):
    data = []
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    image_name_or_path = parts[0]
                    text = '\t'.join(parts[1:])  # Handle cases where text might contain tabs
                    if is_unified:
                        image_path = image_name_or_path
                    else:
                        image_path = os.path.join(image_dir, f"{image_name_or_path}.png")
                    if os.path.exists(image_path):
                        data.append({'image_path': image_path, 'text': text})
    return pd.DataFrame(data)

class HandwrittenDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(row['text'], padding="max_length", max_length=128).input_ids
        labels = torch.tensor(labels)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

def evaluate(model, loader, device, processor):
    model.eval()
    preds = []
    labels_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            pixel_values = batch['pixel_values'].to(device)
            generated_ids = model.generate(pixel_values, max_length=128)
            preds.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))
            label_ids = batch['labels']
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            labels_list.extend(processor.batch_decode(label_ids, skip_special_tokens=True))
    cer = jiwer.cer(labels_list, preds)
    wer = jiwer.wer(labels_list, preds)
    return cer, wer


@torch.no_grad()
def visualize_attention(model, processor, image_path, text, output_dir, prefix, idx, device):
    """
    Visualize TrOCR cross-attention maps over the input image.
    - Each predicted token → one row.
    - Each row shows the original image with that token’s attention overlay.
    """
    font_size=30
    text_margin=20
    os.makedirs(output_dir, exist_ok=True)
    font = ImageFont.load_default()

    # --- 1. Load and preprocess image ---
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values
    resized_h, resized_w = pixel_values.shape[2], pixel_values.shape[3]

    patch_size = model.config.encoder.patch_size
    n_h = resized_h // patch_size
    n_w = resized_w // patch_size
    num_patches = n_h * n_w
    aspect_ratio = resized_w / resized_h

    # --- 2. Run model with attention output ---
    outputs = model.generate(
        pixel_values,
        output_attentions=True,
        return_dict_in_generate=True,
        output_scores=True,
        max_length=128,
    )

    generated_tokens = outputs.sequences[0]
    generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
    print(f"\nGenerated text: {generated_text}")
    print(f"Groundtruth: {text}")

    cross_attentions = outputs.cross_attentions
    num_tokens = len(cross_attentions)
    fig, axes = plt.subplots(num_tokens, 1, figsize=(10, 3 * num_tokens))
    if num_tokens == 1:
        axes = [axes]

    token_texts = processor.tokenizer.convert_ids_to_tokens(generated_tokens)

    # --- 3. For each generated token ---
    attn_imgs = []
    for t_idx in range(num_tokens):
        if t_idx == 0:
            continue 
        layer_attns = cross_attentions[t_idx]

        attn_list = []
        for attn in layer_attns:
            attn_mean = attn.mean(dim=1).squeeze(0)[-1]  # [tgt_len, src_len] - > [src_len]
            attn_list.append(attn_mean)
        attn_map = torch.stack(attn_list).mean(dim=0)  # [src_len]
        # --- 4. Remove CLS token ---
        attn_map = attn_map[1:]  # -> [src_len - 1]

        attn_map = attn_map.reshape(24, 24)
        
        # Upscale attention map to 384x384 (each 1x1 pixel to 16x16)
        attn_map_upscaled = torch.repeat_interleave(attn_map, 16, dim=0)
        attn_map_upscaled = torch.repeat_interleave(attn_map_upscaled, 16, dim=1)
        
        # Convert to numpy and normalize
        attn_map_np = attn_map_upscaled.cpu().numpy()
        attn_map_np = (attn_map_np - attn_map_np.min()) / (attn_map_np.max() - attn_map_np.min() + 1e-8)
        
        # Convert PIL image to numpy array
        original_img = np.array(image)
        orig_h, orig_w = original_img.shape[:2]
        
        # Resize attention map to original image size
        attn_map_pil = Image.fromarray(attn_map_np).resize((orig_w, orig_h), Image.BILINEAR)
        attn_map_np = np.array(attn_map_pil)
        
        # Create blue attention overlay
        attention_overlay = np.zeros_like(original_img)
        attention_overlay[:, :, 2] = (attn_map_np * 255).astype(np.uint8)  # Blue channel
        
        # Blend original image and attention overlay
        blended_img = (0.5 * original_img + 0.5 * attention_overlay).astype(np.uint8)
        
        # Convert back to PIL Image
        blended_img = Image.fromarray(blended_img)

        attn_imgs.append((token_texts[t_idx], blended_img))

    max_text_width = 0
    total_height = 0
    max_img_width = 0
    
    img_widths = []
    img_heights = []
    
    for text, img in attn_imgs:
        img_width, img_height = img.size
        img_widths.append(img_width)
        img_heights.append(img_height)
        max_img_width = max(max_img_width, img_width)
        total_height += img_height
        
        text_width = font.getlength(text) if hasattr(font, 'getlength') else font.getsize(text)[0]
        max_text_width = max(max_text_width, int(text_width))
    
    # Create new image with space for text on the left
    total_width = max_text_width + text_margin + max_img_width
    result_img = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(result_img)
    
    current_y = 0
    
    # Paste each image and draw text
    for (text, img), img_height in zip(attn_imgs, img_heights):
        result_img.paste(img, (max_text_width + text_margin, current_y))
        text_y = current_y + (img_height - font_size) // 2
        draw.text((10, text_y), text, fill=(0, 0, 0), font=font)
        current_y += img_height
    
    # Save the final image
    result_img.save(os.path.join(output_dir, f"{prefix}_sample_{idx}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on handwritten text recognition datasets.")
    parser.add_argument('--dataset', type=str, default='unified', choices=['copiale', 'unified'],
                        help="Dataset to use: 'copiale' for the original Copiale dataset, 'unified' for the combined IAM/CVL/RIMES/EU27 dataset.")
    args = parser.parse_args()

    if args.dataset == 'copiale':
        base_dir = "copiale_gt/"
        image_dir = os.path.join(base_dir, "crop_lines/")
        train_gt = os.path.join(base_dir, "train.gt")
        valid_gt = os.path.join(base_dir, "valid.gt")
        test_gt = os.path.join(base_dir, "test.gt")
        train_df = load_data(train_gt, image_dir=image_dir)
        valid_df = load_data(valid_gt, image_dir=image_dir)
        test_df = load_data(test_gt, image_dir=image_dir)
        best_unified_path = './trocr_unified_best.pt'
        best_model_path = "./trocr_copiale_best.pt"
    elif args.dataset == 'unified':
        unified_gt = "unified_line_iam_cvl_rimes_eu27.txt"
        df = load_data(unified_gt, is_unified=True)
        # Split into train/valid/test (80%/10%/10%)
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        best_model_path = "./trocr_unified_best.pt"

    print(f"Loaded {len(train_df)} training samples, {len(valid_df)} validation samples, {len(test_df)} test samples.")

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    if args.dataset == 'copiale':
        model.load_state_dict(torch.load(best_unified_path))

    train_dataset = HandwrittenDataset(train_df, processor)
    valid_dataset = HandwrittenDataset(valid_df, processor)
    test_dataset = HandwrittenDataset(test_df, processor)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=default_data_collator, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=default_data_collator, num_workers=4)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler('cuda')

    best_cer = math.inf
    patience_count = 0
    os.makedirs('visualizations', exist_ok=True)

    for epoch in range(1, EPOCH+1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch}: Average training loss = {avg_loss}")

        val_cer, val_wer = evaluate(model, valid_loader, device, processor)
        print(f"Epoch {epoch}: Validation CER = {val_cer}, WER = {val_wer}")

        #for i in range(5):
        #    row = train_df.iloc[i]
        #    visualize_attention(model, processor, row['image_path'], row['text'], "./visualizations/train/", "train", i, device)
        #for i in range(5):
        #    row = valid_df.iloc[i]
        #    visualize_attention(model, processor, row['image_path'], row['text'], "./visualizations/valid/", "valid", i, device)

        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), best_model_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}, best epoch at epoch {epoch-PATIENCE}.")
                break

    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.load_state_dict(torch.load(best_model_path))
    model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.to(device)

    train_cer, train_wer = evaluate(model, train_loader, device, processor)
    print(f"Train CER: {train_cer}, WER: {train_wer}")
    valid_cer, valid_wer = evaluate(model, valid_loader, device, processor)
    print(f"Valid CER: {valid_cer}, WER: {valid_wer}")
    test_cer, test_wer = evaluate(model, test_loader, device, processor)
    print(f"Test CER: {test_cer}, WER: {test_wer}")

    for i in range(10):
        row = train_df.iloc[i]
        visualize_attention(model, processor, row['image_path'], row['text'], "./visualizations/train/", "train", i, device)
    for i in range(10):
        row = valid_df.iloc[i]
        visualize_attention(model, processor, row['image_path'], row['text'], "./visualizations/valid/", "valid", i, device)
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        visualize_attention(model, processor, row['image_path'], row['text'], "./visualizations/test/", "test", i, device)

    print("Visualizations saved in ./visualizations/")
