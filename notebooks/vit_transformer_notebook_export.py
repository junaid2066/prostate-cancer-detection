
# %% Cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm
import SimpleITK as sitk
import cv2


# %% [markdown]
# # Read input


# %% Cell 2
import pandas as pd

# Load metadata safely
metadata_path = '/kaggle/input/prostate-cancer-pi-cai-dataset/Metadata with lesion info.csv'
metadata = pd.read_csv(metadata_path)

# Drop unnamed index column if it exists
if 'Unnamed: 0' in metadata.columns:
    metadata.drop(columns=['Unnamed: 0'], inplace=True)

# Ensure IDs are strings (important for consistent merging)
metadata['patient_id'] = metadata['patient_id'].astype(str)
metadata['study_id'] = metadata['study_id'].astype(str)

# Create unique patient-case IDs
metadata['pcID'] = metadata['patient_id'] + "_" + metadata['study_id']

print(f"✅ Metadata loaded successfully: {metadata.shape[0]} records")
display(metadata.head())


# %% [markdown]
# # Define the image processing functions


# %% Cell 4
def display_img(img, title="MRI Slice"):
    """Display the middle slice of a 3D MRI image"""
    img_arr = sitk.GetArrayFromImage(img)
    mid_slice = img_arr.shape[0] // 2
    plt.figure(figsize=(5, 5))
    plt.imshow(img_arr[mid_slice], cmap='gray')
    plt.axis('off')
    plt.title(f'{title} | Slice {mid_slice} | Shape: {img_arr.shape[1:]}')
    plt.show()


def select_slice(img, num_slices):
    """Select centered subset of slices"""
    size = img.GetSize()
    depth = size[2]
    if num_slices >= depth:
        print("⚠️ Requested slices exceed depth — returning full volume.")
        return img
    start_slice = max((depth - num_slices) // 2, 0)
    return sitk.RegionOfInterest(img, [size[0], size[1], num_slices], [0, 0, start_slice])


def zero_pad(img, trg_height, trg_width):
    """Zero-pad to target spatial size"""
    img_arr = sitk.GetArrayFromImage(img)
    h, w = img_arr.shape[1], img_arr.shape[2]
    pad_h = max((trg_height - h) // 2, 0)
    pad_w = max((trg_width - w) // 2, 0)
    padded_img = np.pad(img_arr, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    return sitk.GetImageFromArray(padded_img)


def interpolate(img, trg_height, trg_width):
    """Resize each slice to target dimensions"""
    img_arr = sitk.GetArrayFromImage(img)
    resized_slices = [cv2.resize(slice, (trg_width, trg_height), interpolation=cv2.INTER_CUBIC)
                      for slice in img_arr]
    resized_arr = np.stack(resized_slices, axis=0)
    return sitk.GetImageFromArray(resized_arr)


def normalize(img_arr):
    """Normalize image to [0, 1]"""
    img_max = img_arr.max()
    return img_arr / (img_max + 1e-8)


def standardization(image):
    """Z-score standardization"""
    mean, std = np.mean(image), np.std(image)
    return (image - mean) / std if std != 0 else image



# %% [markdown]
# # Extract the images from dataset


# %% Cell 6
import os, shutil

root_dir = '/kaggle/input/prostate-cancer-pi-cai-dataset'
output_dir = '/kaggle/working/output_directory'
os.makedirs(output_dir, exist_ok=True)

num_cases = 0
copied_cases = set()

for main_folder in os.listdir(root_dir):
    main_folder_path = os.path.join(root_dir, main_folder)
    
    if os.path.isdir(main_folder_path):
        for patient_folder in os.listdir(main_folder_path):
            patient_folder_path = os.path.join(main_folder_path, patient_folder)
            
            if os.path.isdir(patient_folder_path):
                for file_name in os.listdir(patient_folder_path):
                    # Extract patient_case_id (first two parts or first three if needed)
                    parts = file_name.split('_')
                    if len(parts) >= 2:
                        patient_case_id = f"{parts[0]}_{parts[1]}"
                        # Optionally include the 3rd part if that’s how IDs appear in metadata
                        if patient_case_id not in pcID and len(parts) >= 3:
                            patient_case_id = f"{parts[0]}_{parts[1]}_{parts[2]}"

                        # Match with metadata IDs
                        if patient_case_id in pcID:
                            output_subdir = os.path.join(output_dir, patient_case_id)
                            os.makedirs(output_subdir, exist_ok=True)

                            # Copy only axial (skip sagittal/coronal)
                            if not (file_name.endswith('_sag.mha') or file_name.endswith('_cor.mha')):
                                src = os.path.join(patient_folder_path, file_name)
                                dst = os.path.join(output_subdir, file_name)
                                shutil.copy(src, dst)

                                if patient_case_id not in copied_cases:
                                    copied_cases.add(patient_case_id)
                                    num_cases += 1

print("✅ Total cases copied:", num_cases)
print("✅ Unique patient-case IDs found:", len(copied_cases))



# %% Cell 7
for case_folder in os.listdir(output_dir):
    case_folder_path = os.path.join(output_dir,case_folder)
    
    if os.path.isdir(case_folder_path): 
        for file_name in os.listdir(case_folder_path):
            file_path = os.path.join(case_folder_path,file_name)
            img = sitk.ReadImage(file_path)
            resized_img = interpolate(select_slice(img,16),224,224)
            sitk.WriteImage(resized_img, file_path)


metadata['PatientID_CaseID'] = metadata['patient_id'].astype(str) + "_" + metadata['study_id'].astype(str)
metadata.head()


# %% [markdown]
# # Apply augmentation


# %% Cell 9
import albumentations as A
import numpy as np
import os
import SimpleITK as sitk
import pandas as pd

# Augmentation mapping per ISUP grade
aug_grades = {3: 1, 4: 5, 5: 3}  # ISUP grade → number of augmentations

# Augmentation pipeline (based on Table 2)
medical_aug= A.Compose([
    A.Rotate(limit=45, p=0.5),  # Rotation: 0–45°
    A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, p=0.5),  # Shift & Zoom
    A.Affine(shear={'x': (-25, 25)}, p=0.3), 
    A.Affine(shear={'y': (-25, 25)}, p=0.3), 
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(0.5, 2.0), contrast_limit=0.0, p=0.4),
])

# Apply augmentations to a single 2D slice
def apply_augmentations(image_slice):
    image_slice = np.clip(image_slice, 0, None)  # Remove negatives
    augmented = medical_aug(image=image_slice)['image']
    return augmented

# Loop through metadata
for idx, row in metadata.iterrows():
    grade = row['case_ISUP']
    if grade not in aug_grades:
        continue

    case_id = row['PatientID_CaseID']
    case_folder = os.path.join(output_dir, case_id)

    if not os.path.exists(case_folder):
        continue

    # Read original images once
    original_images = {}
    for modality in ['t2w', 'adc', 'hbv']:
        original_file = os.path.join(case_folder, f"{case_id}_{modality}.mha")
        if os.path.exists(original_file):
            original_images[modality] = sitk.ReadImage(original_file)

    # Apply augmentations
    for i in range(aug_grades[grade]):
        aug_folder = os.path.join(output_dir, f"{case_id}_aug_{i+1:03d}")
        os.makedirs(aug_folder, exist_ok=True)

        for modality, image in original_images.items():
            image_array = sitk.GetArrayFromImage(image)  # Shape: (D, H, W)
            D, H, W = image_array.shape

            augmented_slices = []
            for d in range(D):
                slice_2d = image_array[d]
                augmented_slice = apply_augmentations(slice_2d)
                augmented_slices.append(augmented_slice)

            # Reconstruct 3D image
            aug_sitk = sitk.GetImageFromArray(np.array(augmented_slices))
            aug_sitk.CopyInformation(image)

            # Save file
            aug_file = os.path.join(aug_folder, f"{case_id}_aug_{i+1:03d}_{modality}.mha")
            sitk.WriteImage(aug_sitk, aug_file)



# %% Cell 10
image_data = []

# Iterate through patient folders, including augmented ones
for patient_case in os.listdir(output_dir):
    patient_dir = os.path.join(output_dir, patient_case)
    
    # Skip if not a directory
    if not os.path.isdir(patient_dir):
        continue
    
    # Extract the original case ID (handles both original & augmented cases)
    base_case_id = patient_case.split('_aug_')[0]  # Extract original case ID
    
    # Check if base case ID exists in the metadata
    if base_case_id in metadata['PatientID_CaseID'].values:
        # Fetch metadata for this patient
        case_data = metadata.loc[metadata['PatientID_CaseID'] == base_case_id].iloc[0]
        
        # Iterate through files in the case folder
        for file_name in os.listdir(patient_dir):
            file_path = os.path.join(patient_dir, file_name)
            
            if file_name.endswith('.mha'):
                # Extract modality (adc, hbv, t2w)
                modality = file_name.split('_')[-1].split('.')[0]
                
                # Determine if it's an augmented image
                is_augmented = "_aug_" in patient_case  # Check if it's augmented
                aug_suffix = ""
                
                if is_augmented:
                    # Assign a unique augmented name (10005_100005_a1, a2, etc.)
                    aug_id = patient_case.split('_aug_')[-1]  # Extract augmentation number
                    patient_case_name = f"{base_case_id}_a{int(aug_id):02d}"
                else:
                    patient_case_name = base_case_id  # Keep original ID for non-augmented data

                # Append mapping information
                image_data.append({
                    'ImagePath': file_path,
                    'PatientID_CaseID': patient_case_name,  # Modified for augmented cases
                    'Augmented': is_augmented,  # True if it's an augmented image
                    'Modality': modality,
                    'psa': case_data['psa'],
                    'psad': case_data['psad'],
                    'prostate_volume': case_data['prostate_volume'],
                    'num_lesions': case_data['num_lesions'],
                    'max_prim_score': case_data['max_prim_score'],
                    'max_sec_score': case_data['max_sec_score'],
                    'max_gleason_scores': case_data['max_gleason_scores'],
                    'case_ISUP': case_data['case_ISUP'],
                    'case_csPCa': case_data['case_csPCa']
                })

# Convert list to DataFrame
image_df = pd.DataFrame(image_data)
image_df.head(10)


# %% Cell 11
image_df.shape
round(image_df['case_csPCa'].value_counts()/3)
round(image_df['case_ISUP'].value_counts()/3)


# %% [markdown]
# # Feature extraction


# %% Cell 13
from transformers import ViTFeatureExtractor, TFAutoModel
from PIL import Image
import math
import gc

vit_model = TFAutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

def extract_vit_features(img_path, csv_file, is_first_entry=False):
    try:
        # Load image with SimpleITK and convert to RGB
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img)
        if len(img_array.shape) == 3:
            img_array = np.transpose(img_array, (1, 2, 0))
            img_array = img_array[:, :, :3]

        img_array = img_array.astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        # Preprocess
        inputs = feature_extractor(images=pil_img, return_tensors="tf")
        outputs = vit_model(**inputs)

        # Take [CLS] token as the feature vector
        cls_token = outputs.last_hidden_state[:, 0, :].numpy().flatten().tolist()

        df = pd.DataFrame([[img_path, cls_token]], columns=['ImagePath', 'Features'])
        df.to_csv(csv_file, mode='a', header=is_first_entry, index=False)

        del img, img_array, pil_img, inputs, outputs, cls_token, df
        gc.collect()

    except Exception as e:
        print(f"Error processing {img_path}: {e}")


def process_images(image_df, batch_size, output_csv):
    batch_count = 0
    total_batches = math.ceil(len(image_df) / batch_size)

    # Ensure CSV is empty before writing
    if os.path.exists(output_csv):
        os.remove(output_csv)

    for start_idx in range(0, len(image_df), batch_size):
        end_idx = start_idx + batch_size
        batch_df = image_df.iloc[start_idx:end_idx]

        for idx, row in batch_df.iterrows():
            img_path = row['ImagePath']
            is_first_entry = (batch_count == 0 and idx == start_idx)  # True only for the first row of first batch
            extract_vit_features(img_path, output_csv, is_first_entry)

        batch_count += 1
        print(f"Completed {batch_count}/{total_batches} batches")

        gc.collect()  


batch_size = 8
output_csv = '/kaggle/working/image_features.csv'

process_images(image_df, batch_size, output_csv)

features_df = pd.read_csv(output_csv)
print("Feature extraction complete!")




# %% [markdown]
# # Make the final metadata


# %% Cell 15
img_features_df = pd.merge(image_df,features_df,how = 'inner',on='ImagePath')
img_features_df["Features"] = img_features_df["Features"].apply(lambda x: np.array(x))
img_features_df.head()


# %% Cell 16
import ast
img_features_df.shape
img_features_df.info()
def convert(x):
    """If the feature vector is written as a string to the csv, then convert it to numpy array"""
    if isinstance(x, np.ndarray):  
        return x
    elif isinstance(x, str):  
        try:
            return np.array(ast.literal_eval(x), dtype=np.float16)
        except (ValueError, SyntaxError):  
            return None  
    else:
        return None  

img_features_df['Features'] = img_features_df['Features'].apply(convert)

def concatenate_features(df):
    """Stacks the feature vector of all modalities for a particular case"""
    grouped = df.groupby('PatientID_CaseID')
    concatenated_data = []

    for patient_id, patient_data in grouped:
        features = [f for f in patient_data['Features'].values]
        concatenated_features = np.hstack(features) if features else None    
        concatenated_data.append({
            'PatientID_CaseID': patient_id,
            'Concatenated_Features': concatenated_features})

    return pd.DataFrame(concatenated_data)


concatenated_features_df = concatenate_features(img_features_df)


columns_to_keep = [col for col in img_features_df.columns if col not in ['Modality','Features', 'Augmented', 'ImagePath']]
image_metadata_df = img_features_df[columns_to_keep].drop_duplicates()


metadata_final = pd.merge(image_metadata_df, concatenated_features_df, how='inner', on='PatientID_CaseID')


metadata_final['PatientID_CaseID'] = metadata_final['PatientID_CaseID'].apply(
    lambda x: x if '_aug_' not in x else f"{x}_a1"
)

metadata_final.head(10)


# %% Cell 17
metadata_final.info()
metadata_final.shape
metadata_final['Concatenated_Features'][0].shape
feature_lengths = metadata_final['Concatenated_Features'].apply(lambda x: len(x))
print("Unique lengths of feature vectors:", feature_lengths.unique())


# %% Cell 18
feature_lengths = metadata_final['Concatenated_Features'].apply(lambda x: len(x))
length_counts = feature_lengths.value_counts()
print(length_counts)


# %% Cell 19
# Assuming 'patientID_caseID' is the column name for patient-case identifiers
incorrect_entries = metadata_final[metadata_final['Concatenated_Features'].apply(lambda x: len(x) != 2304)]

# Display the patientID_caseID of the incorrect entry
print(incorrect_entries[['PatientID_CaseID', 'Concatenated_Features']])


# %% Cell 20
metadata_final = metadata_final[metadata_final['Concatenated_Features'].apply(lambda x: len(x) == 2304)].reset_index(drop=True)

print("Filtered dataset shape:", metadata_final.shape)


# %% Cell 21
feature_lengths = metadata_final['Concatenated_Features'].apply(lambda x: len(x))
length_counts = feature_lengths.value_counts()
print(length_counts)


# %% [markdown]
# # Train the model


# %% Cell 23
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf


# -------- Data preparation --------
feature_vector = np.array(metadata_final['Concatenated_Features'].tolist())
sc = StandardScaler()
feature_vector_scaled = sc.fit_transform(feature_vector)

metadata_columns = ['psa', 'psad', 'prostate_volume','num_lesions','max_prim_score','max_sec_score','max_gleason_scores']
metadata_values = metadata_final[metadata_columns].values
metadata_scaled = sc.fit_transform(metadata_values)

x = np.hstack((metadata_scaled, feature_vector_scaled)) 

y = metadata_final['case_csPCa'].values 
y = LabelEncoder().fit_transform(y)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold = 1
acc_scores = []
auc_scores = []

for train_index, val_index in skf.split(x, y):
    print(f"\n Fold {fold}")

    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    model = Sequential([
    Input(shape=(x.shape[1],)),
    BatchNormalization(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.48),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])


    model.compile(optimizer=Adam(learning_rate=0.05),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=36,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=0
    )

    val_loss, val_acc,val_auc = model.evaluate(x_val, y_val, verbose=0)
    y_pred_proba = model.predict(x_val, verbose=0)

    print(f" Fold {fold} Accuracy: {val_acc:.4f} | AUC: {val_auc:.4f}")

    acc_scores.append(val_acc)
    auc_scores.append(val_auc)
    fold += 1

# -------- Final report --------
print("\n Cross-Validation Results:")
print(f"Average Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
print(f"Average AUC:      {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")



# %% Cell 24
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import warnings
from tensorflow.keras import regularizers
warnings.filterwarnings('ignore')


feature_vector = np.array(metadata_final['Concatenated_Features'].tolist())
sc = StandardScaler()
feature_vector_scaled = sc.fit_transform(feature_vector)
joblib.dump(sc, 'scaler.pkl')
print("✅ Scaler saved as 'scaler.pkl'")

metadata_columns = ['psa', 'psad', 'prostate_volume','num_lesions','max_prim_score','max_sec_score','max_gleason_scores']
metadata_values = metadata_final[metadata_columns].values
metadata_scaled = sc.fit_transform(metadata_values)

x = np.hstack((metadata_scaled, feature_vector_scaled)) 

y = metadata_final[['case_ISUP','case_csPCa']].values
le = LabelEncoder()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True, random_state=42, stratify=y[:,1])

case_ISUP = y_test[:,0]
case_csPCa = y_test[:,1]

y_train = y_train[:,1]
y_test = y_test[:,1]

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))


model = Sequential([
    Input(shape=(x.shape[1],)),
    BatchNormalization(),

    Dense(256, kernel_regularizer=regularizers.l2(0.001)),
    LeakyReLU(),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, kernel_regularizer=regularizers.l2(0.001)),
    LeakyReLU(),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32),
    LeakyReLU(),
    BatchNormalization(),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

callbacks = [
    EarlyStopping(monitor='val_auc', patience=7, restore_best_weights=True, verbose=1, mode='max'),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, min_lr=1e-6, verbose=1, mode='max')
]

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

train_loss, train_acc, train_auc = model.evaluate(x_train, y_train, verbose=0)
print(f"\n✅ Train Accuracy: {train_acc:.4f} | Train AUC: {train_auc:.4f}")





# %% [markdown]
# # Testing the model


# %% Cell 26
from sklearn.metrics import classification_report, roc_auc_score

y_probs = model.predict(x_test)
y_pred = (y_probs >= 0.5).astype(int)
print("Classification Report:\n", classification_report(y_test, y_pred))
final_auc = roc_auc_score(y_test, y_probs)
print(f"\n Final Test AUC: {final_auc:.4f}")
loss = model.evaluate(x_test, y_test, verbose=0)[0]
print(f"Test Loss: {loss:.4f}")



# %% [markdown]
# # Confusion matrix


# %% Cell 28
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = (y_probs >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
