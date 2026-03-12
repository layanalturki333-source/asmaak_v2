# Dataset Guide (Prototype)

This guide describes how to prepare video data for the 5-word sign language prototype.

## Vocabulary

| Word (ID) | Arabic Label |
|-----------|--------------|
| hello     | مرحبا       |
| yes       | نعم         |
| no        | لا          |
| thank_you | شكرا        |
| water     | ماء         |

## Folder Structure

Place your videos under the `dataset/` directory, one subfolder per word:

```
dataset/
  hello/
    video_001.mp4
    video_002.mp4
    ...
  yes/
    video_001.mp4
    ...
  no/
    ...
  thank_you/
    ...
  water/
    ...
```

## Requirements

- **Format**: Supported extensions are `.mp4`, `.avi`, `.mov`, `.mkv` (OpenCV must be able to read the file).
- **Filenames**: Any filename is accepted (e.g. `video_001.mp4`, `Movie on 12-03-2026 at 3.52 AM.mov`). No specific naming pattern is required.
- **Content**: One sign per video (isolated sign). Hand(s) visible and well lit.
- **Quantity**: At least 5–10 videos per word for a minimal trainable dataset; more is better.

## Steps

1. **Record or collect videos** for each of the 5 words.
2. **Organize** them into `dataset/<word>/` as above. Word names must match exactly: `hello`, `yes`, `no`, `thank_you`, `water`.
3. **Run landmark extraction** (see README):
   ```bash
   python src/extract_landmarks.py
   ```
   This reads from `dataset/` and writes sequences to `data/` (e.g. `train_sequences.npy`, `train_labels.npy`, `labels.json`).
4. **Train the model**:
   ```bash
   python src/train.py
   ```
   This reads from `data/` and saves the model under `models/`.

## Labels File

After extraction, `data/labels.json` will contain the list of label strings in index order, e.g.:

```json
["hello", "yes", "no", "thank_you", "water"]
```

The UI uses the Arabic labels from `vocabulary_sheet.csv` for display.
