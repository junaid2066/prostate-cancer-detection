
# %% Cell 0
# Import necessary libraries
import os
import gc
import cv2
import ast
import math
import shutil
import numpy as np
import pandas as pd 
import albumentations as A
from PIL import Image
import SimpleITK as sitk
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import precision_score, recall_score, roc_auc_score,classification_report,confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from transformers import AutoFeatureExtractor, TFAutoModel
import warnings
warnings.filterwarnings('ignore')


# %% Cell 1
# Read the metadata file
metadata = pd.read_csv('/kaggle/input/prostate-cancer-pi-cai-dataset/Metadata with lesion info.csv')


# %% Cell 2
metadata.info()


# %% Cell 3
metadata.head()


# %% Cell 4
metadata.drop(columns = ['Unnamed: 0'],inplace = True)


# %% Cell 5
# Create patientID_caseID for each record in the metadata

patientID = list(metadata['patient_id'])
caseID = list(metadata['study_id'])

patientID = [str(i) for i in patientID]
caseID = [str(i) for i in caseID]

pcID = []
for i,j in zip(patientID,caseID):
    pcID.append(i+'_'+j)
#metadata['patient_caseID'] = pcID


# %% [markdown]
# ### Helper functions:
# 1) To display a slice of the MRI<br>
# 2) Select the no. of slices to consider from the MRI<br>
# 3) Resize to specific (height,width) using zero padding (or) bicublic interpolation


# %% Cell 7
def display_img(img,title):
    '''Displays the middle slice of the MRI with it's size'''
    img_arr = sitk.GetArrayFromImage(img)
    plt.imshow(img_arr[img_arr.shape[0]//2],cmap = 'gray') # Middle slice of the image
    plt.axis('off')
    plt.title(f'{title}: {img_arr.shape[1:]}')
    plt.show()


# %% Cell 8
def select_slice(img,num_slices):
    '''Takes the specified number of slices from the middle portion of the original image''' 
    size = img.GetSize()
    depth = size[2]
    start_slice = (depth - num_slices)//2
    end_slice = start_slice + num_slices
    return sitk.RegionOfInterest(img,[size[0],size[1],num_slices],[0,0,start_slice])
   


# %% Cell 9
def zero_pad(img,trg_height,trg_width):
    '''Pads the given image to required height & width'''

    # Convert the sitk object into a numpy array
    img_arr = sitk.GetArrayFromImage(img)
    
    # Calculate the remaining area to be padded
    pad_height = (trg_height - img_arr.shape[1])//2 # Gives the size for padding top & bottom
    pad_width = (trg_width - img_arr.shape[2])//2 # Gives the size for padding left & right

    # Ensure padding is done equally on all sides
    pad_3d = ((0,0),(pad_height,pad_height),(pad_width,pad_width))

    # Applying zero padding across all sides
    padded_img = np.pad(img_arr,pad_3d,mode = 'constant',constant_values = 0)

    return sitk.GetImageFromArray(padded_img)


# %% Cell 10
def interpolate(img,trg_height,trg_width):
    '''Performs bicubic interpolation slice by slice'''
    img_arr = sitk.GetArrayFromImage(img)

    # Contains the newly resized slices
    resized_slices = []

    for i in range(img_arr.shape[0]):
        # Resize each slice using bicubic interpolation
        resized_slices.append(cv2.resize(img_arr[i],(trg_height,trg_width),interpolation = cv2.INTER_CUBIC))

    # Stack the resized slices
    resized_arr = np.stack(resized_slices, axis = 0)
    
    # Return normalized image
    return sitk.GetImageFromArray(resized_arr)


# %% [markdown]
# ### Move the necessary files to output directory
# Each folder is named after the patientID, each consists of 5 MRI modalities.<br>
# We consider only the patients whose details exist in the preprocessed metadata file.<br>
# We rename the folders as patient_caseID where each folder will contain only 3 MRI modalities(ADC, HBV, T2W) since a patient can have more than 1 case


# %% Cell 12
# Defining the root directory
root_dir = '/kaggle/input/prostate-cancer-pi-cai-dataset'

# Creating an output directory
output_dir = '/kaggle/working/output_directory'
os.makedirs(output_dir, exist_ok=True)


# %% Cell 13
# To ensure no. of records in the metadata match no. of cases in the output directory
num_cases = 0

# Iterate through the 5 main folders under the root directory
for main_folder in os.listdir(root_dir):
    main_folder_path = os.path.join(root_dir, main_folder)
    
    if os.path.isdir(main_folder_path):
        # Iterate through patient subfolders in each of the 5 main folders
        for patient_folder in os.listdir(main_folder_path):
            patient_folder_path = os.path.join(main_folder_path, patient_folder)
            
            if os.path.isdir(patient_folder_path):
                # Iterate through files in the patient folder
                for file_name in os.listdir(patient_folder_path):
                    # Extract patientID_caseID from the filename
                    patient_case_id = str(file_name.split('_')[0]+'_'+file_name.split('_')[1]) 
                    
                    if patient_case_id in pcID:
                            # Create a corresponding folder for each case in the output directory
                            output_subdir = os.path.join(output_dir, patient_case_id)
                            if not os.path.exists(output_subdir):  # Check if the folder exists
                                os.makedirs(output_subdir, exist_ok=True)
                                num_cases += 1
                                
                            # Exclude files ending with '_sag.mha' or '_cor.mha'
                            if not (file_name.endswith('_sag.mha') or file_name.endswith('_cor.mha')):
                                file_path = os.path.join(patient_folder_path, file_name)
                                shutil.copy(file_path, output_subdir)

print("Total cases:",num_cases)


# %% [markdown]
# ### Resizing MRIs
# We consider only the middle 16 slices of the MRIs.<br>
# All MRI files are resized to (16,224,224) by performing bicubic interpolation.


# %% Cell 15
for case_folder in os.listdir(output_dir):
    case_folder_path = os.path.join(output_dir, case_folder)
    
    if os.path.isdir(case_folder_path): 
        for file_name in os.listdir(case_folder_path):
            file_path = os.path.join(case_folder_path, file_name)
            img = sitk.ReadImage(file_path)
            resized_img = interpolate(select_slice(img,16),224,224)  
            sitk.WriteImage(resized_img, file_path)  # Overwriting the original file


# %% Cell 16
# Create a unique identifier for each case
metadata['PatientID_CaseID'] = metadata['patient_id'].astype(str) + "_" + metadata['study_id'].astype(str)


# %% Cell 17
metadata.head()


# %% Cell 18
 # No of augmented images to generate based on grade
aug_grades = {3: 1, 4: 5, 5: 3} 


medical_aug = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(blur_limit=(3,5), p=0.2),
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    A.RandomGamma(gamma_limit=(80, 120), p=0.2),
])


def apply_augmentations(image_slice):
    image_slice = np.clip(image_slice, 0, None)  # Clip negative values
    augmented = medical_aug(image=image_slice)['image']
    return augmented

# Iterate through metadata
for idx, row in metadata.iterrows():
    grade = row['case_ISUP']
    if grade not in aug_grades:
        continue
        
    case_id = row['PatientID_CaseID']
    case_folder = os.path.join(output_dir, case_id)
    
    if os.path.exists(case_folder):
        original_output_folder = os.path.join(output_dir, case_id)
        os.makedirs(original_output_folder, exist_ok=True)
        
        original_images = {}  # Store original images

        for modality in ['t2w', 'adc', 'hbv']:
            original_file = os.path.join(case_folder, f"{case_id}_{modality}.mha")
            if os.path.exists(original_file):
                output_file = os.path.join(original_output_folder, f"{case_id}_{modality}.mha")
                original_images[modality] = sitk.ReadImage(original_file)
                sitk.WriteImage(original_images[modality], output_file)
                
                # Get image array and process each slice separately
                image_array = sitk.GetArrayFromImage(original_images[modality])  # Shape: (D, H, W)
                D, H, W = image_array.shape  

                for i in range(aug_grades[grade]):
                    aug_folder = os.path.join(output_dir, f"{case_id}_aug_{i+1:03d}")
                    os.makedirs(aug_folder, exist_ok=True)
                    
                    augmented_slices = []
                    
                    for d in range(D):  # Process each slice individually
                        slice_2d = image_array[d]  # Shape: (H, W)
                        augmented_slice = apply_augmentations(slice_2d) 
                        augmented_slices.append(augmented_slice)

                    # Convert back to 3D image
                    aug_sitk = sitk.GetImageFromArray(np.array(augmented_slices))  # Shape: (D, H, W)
                    aug_sitk.CopyInformation(original_images[modality])
                    
                    aug_file = os.path.join(aug_folder, f"{case_id}_aug_{i+1:03d}_{modality}.mha")
                    sitk.WriteImage(aug_sitk, aug_file)


# %% Cell 19
# Initialize a list to store mapping information
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




# %% Cell 20
image_df.shape


# %% Cell 21
round(image_df['case_csPCa'].value_counts()/3)


# %% Cell 22
round(image_df['case_ISUP'].value_counts()/3)


# %% [markdown]
# ### Feature extraction using Swin Transformers


# %% Cell 24
# Install required libraries
#!pip install tensorflow tensorflow-hub


# %% Cell 25
#!pip install transformers tensorflow torch pandas numpy SimpleITK


# %% Cell 26
model_name = "microsoft/swin-base-patch4-window7-224" 
# Here, patch4 - Splits image into 4*4 patches, window7 - Has a sliding window of 7*7 for attention
# 224 - Takes an input of size 224*224

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name) # Loads the feature extractor, it converts these images to tensors
model = TFAutoModel.from_pretrained(model_name)

# Saving the model config & the feature extractor settings locally
model.save_pretrained("swin_base_model") 
feature_extractor.save_pretrained("swin_base_feature_extractor")


# %% Cell 27
def normalize_image(img_array):
    """ Normalize images to [0, 255] range for feature extraction """
    min_val = img_array.min()
    max_val = img_array.max()
    
    # Normalize to [0, 255]
    if max_val > 255:
        img_array = ((img_array - min_val) / (max_val - min_val)) * 255
    
    # Ensure uint8 type
    img_array = img_array.astype(np.uint8)
    return img_array
    
def extract_features(img_path, feature_extractor, model, csv_file, is_first_entry=False):
    """ Extract features from the MRIs & write it to a csv file """
    try:
        img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img)
        
        # Ensure the image is 3D
        if len(img_array.shape) == 3:
            img_array = np.transpose(img_array, (1, 2, 0))  # Convert (C, H, W) to (H, W, C)
            img_array = img_array[:, :, :3]  # Keep only 3 channels 

        img_array = normalize_image(img_array)
        
        inputs = feature_extractor(
            img_array, 
            return_tensors="tf", 
            do_rescale=True # Rescaling pixel values to [0,1]
        )
        
        outputs = model(inputs, training=False)
        features_np = outputs.last_hidden_state[:, 0, :].numpy().flatten() # Converts feature vector tensor to 1D numpy array

        df = pd.DataFrame([[img_path, features_np.tolist()]], columns=['ImagePath', 'Features'])
        df.to_csv(csv_file, mode='a', header=is_first_entry, index=False)
        
        del img, img_array, inputs, outputs, features_np, df
        gc.collect()
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def process_images(image_df, feature_extractor, model, batch_size, output_csv):
    """Process images in batches, extract feature vectors & write to a csv"""
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    # Keeping track of total batches
    batch_count = 0
    total_batches = math.ceil(len(image_df) / batch_size)
    
    # Process images in batches
    for start_idx in range(0, len(image_df), batch_size):
        end_idx = start_idx + batch_size
        batch_df = image_df.iloc[start_idx:end_idx]
        
        for idx, row in batch_df.iterrows():
            img_path = row['ImagePath']
            is_first_entry = (batch_count == 0 and idx == start_idx)

            # Performing feature extraction & writing it to the csv file
            extract_features(img_path, feature_extractor, model, output_csv, is_first_entry)
        
        batch_count += 1
        print(f"Completed {batch_count}/{total_batches} batches")
        gc.collect()


output_csv = "image_features.csv"
batch_size = 8

# Process images in batches of 8, extract their features & write to a csv
process_images(image_df,feature_extractor, model, batch_size, output_csv)
print("Feature extraction complete!")


# %% Cell 28
import os
import pandas as pd

# Path to extracted Swin Transformer features
features_path = "/kaggle/working/image_features.csv"

# --- Step 1: Check if the file exists ---
if not os.path.exists(features_path):
    raise FileNotFoundError(f"❌ File not found at: {features_path}\n"
                            "Please ensure the feature extraction step completed successfully.")

# --- Step 2: Read the features file ---
features_df = pd.read_csv(features_path)

# --- Step 3: Validate and preview ---
if features_df.empty:
    raise ValueError("❌ The features file is empty. Re-check your extraction pipeline.")

print("✅ Features loaded successfully!")
print(f"📂 Path: {features_path}")
print(f"🧮 Shape: {features_df.shape[0]} samples × {features_df.shape[1]} columns")
print("\n🧩 Columns:", features_df.columns.tolist())

# --- Step 4: Display top rows ---
display(features_df.head())


# %% Cell 29
import numpy as np
import pandas as pd
import ast

# --- Step 1: Merge image metadata and extracted features ---
img_features_df = pd.merge(
    image_df, 
    features_df, 
    how='inner', 
    on='ImagePath'
)

print(f"✅ Merged successfully — Shape: {img_features_df.shape}")
print(f"🧩 Columns: {img_features_df.columns.tolist()}")

# --- Step 2: Convert stringified features to NumPy arrays safely ---
def parse_feature(x):
    try:
        # Convert string representation of list to real list
        if isinstance(x, str):
            x = ast.literal_eval(x)
        # Convert to NumPy array
        return np.array(x, dtype=np.float32)
    except Exception as e:
        print(f"⚠️ Failed to parse feature: {e}")
        return np.array([])

img_features_df["Features"] = img_features_df["Features"].apply(parse_feature)

# --- Step 3: Drop rows with empty or invalid feature vectors ---
initial_len = len(img_features_df)
img_features_df = img_features_df[img_features_df["Features"].apply(lambda x: x.size > 0)]
removed = initial_len - len(img_features_df)
if removed > 0:
    print(f"⚠️ Removed {removed} rows with invalid feature data.")

# --- Step 4: Verify feature consistency ---
feature_shapes = img_features_df["Features"].apply(lambda x: x.shape[0]).value_counts()
print(f"✅ Feature vector length distribution:\n{feature_shapes}")

# --- Step 5: Display a few samples ---
display(img_features_df.head())



# %% Cell 30
img_features_df.shape


# %% Cell 31
img_features_df.info()


# %% [markdown]
# # Concatenation of Features


# %% Cell 33
import ast
import numpy as np

def convert_feature(x):
    """
    Safely converts the 'Features' column values into NumPy arrays.
    - Handles both already-converted arrays and stringified lists.
    - Returns None for invalid or unparsable values.
    """
    if isinstance(x, np.ndarray):
        # Already an array — no action needed
        return x
    
    elif isinstance(x, str):
        try:
            # Convert string like "[0.12, -0.45, ...]" → list → np.array
            arr = np.array(ast.literal_eval(x), dtype=np.float32)
            
            # Ensure it's a valid 1D feature vector (not empty or malformed)
            if arr.ndim == 1 and arr.size > 0:
                return arr
            else:
                print("⚠️ Skipping malformed feature vector.")
                return None
                
        except (ValueError, SyntaxError):
            print("⚠️ Could not parse feature string.")
            return None

    # For anything else (NaN, None, etc.)
    else:
        return None


# --- Apply conversion function ---
img_features_df['Features'] = img_features_df['Features'].apply(convert_feature)

# --- Drop invalid or empty features ---
initial_len = len(img_features_df)
img_features_df = img_features_df.dropna(subset=['Features'])
img_features_df = img_features_df[img_features_df['Features'].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)]
removed = initial_len - len(img_features_df)

print(f"✅ Features converted successfully — Remaining samples: {len(img_features_df)}")
if removed > 0:
    print(f"⚠️ Removed {removed} invalid feature rows.")

# --- Sanity check ---
feature_shapes = img_features_df['Features'].apply(lambda x: x.shape[0]).value_counts()
print("🧩 Feature vector length distribution:\n", feature_shapes)

display(img_features_df.head())



# %% Cell 34
img_features_df['Features'][0].shape


# %% Cell 35
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


# %% Cell 36
concatenated_features_df = concatenate_features(img_features_df)


columns_to_keep = [col for col in img_features_df.columns if col not in ['Modality','Features', 'Augmented', 'ImagePath']]
image_metadata_df = img_features_df[columns_to_keep].drop_duplicates()


metadata_final = pd.merge(image_metadata_df, concatenated_features_df, how='inner', on='PatientID_CaseID')


metadata_final['PatientID_CaseID'] = metadata_final['PatientID_CaseID'].apply(
    lambda x: x if '_aug_' not in x else f"{x}_a1"
)

metadata_final.head(10)


# %% Cell 37
metadata_final.info()


# %% Cell 38
metadata_final.shape


# %% Cell 39
metadata_final['Concatenated_Features'][0].shape


# %% Cell 40
# Check the lengths of feature vectors
feature_lengths = metadata_final['Concatenated_Features'].apply(lambda x: len(x))
print("Unique lengths of feature vectors:", feature_lengths.unique())



# %% Cell 41
feature_lengths = metadata_final['Concatenated_Features'].apply(lambda x: len(x))
length_counts = feature_lengths.value_counts()
print(length_counts)



# %% Cell 42
feature_vector = np.vstack(metadata_final['Concatenated_Features'].values)
print("✅ Feature matrix shape:", feature_vector.shape)


# %% Cell 43
# --- Step 7: Feature scaling ---
scaler = StandardScaler()
feature_vector_scaled = scaler.fit_transform(feature_vector)
print("✅ Scaled feature vector shape:", feature_vector_scaled.shape)


# %% Cell 44
# --- Step 8: Save results ---
metadata_final['Scaled_Features'] = feature_vector_scaled.tolist()
metadata_final.to_csv("/kaggle/working/metadata_features_ready.csv", index=False)
print("✅ Saved merged and scaled features successfully.")


# %% Cell 45
# Assuming 'patientID_caseID' is the column name for patient-case identifiers
incorrect_entries = metadata_final[metadata_final['Concatenated_Features'].apply(lambda x: len(x) != 3072)]

# Display the patientID_caseID of the incorrect entry
print(incorrect_entries[['PatientID_CaseID', 'Concatenated_Features']])



# %% Cell 46
# Keep only rows where the feature vector length is 6144
metadata_final = metadata_final[metadata_final['Concatenated_Features'].apply(lambda x: len(x) == 6144)].reset_index(drop=True)

print("Filtered dataset shape:", metadata_final.shape)


# %% Cell 47
feature_lengths = metadata_final['Concatenated_Features'].apply(lambda x: len(x))
length_counts = feature_lengths.value_counts()
print(length_counts)



# %% [markdown]
# ### Model building (including metadata for classification)


# %% Cell 49
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ast

# --- Step 1: Verify the column exists ---
if 'Concatenated_Features' not in metadata_final.columns:
    raise KeyError("Column 'Concatenated_Features' not found in metadata_final DataFrame.")

# --- Step 2: Safely parse feature lists (in case they are stored as strings) ---
def safe_parse(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    elif isinstance(x, (list, np.ndarray)):
        return x
    else:
        return []

metadata_final['Concatenated_Features'] = metadata_final['Concatenated_Features'].apply(safe_parse)

# --- Step 3: Convert to NumPy array ---
feature_vector = np.array(metadata_final['Concatenated_Features'].tolist(), dtype=object)

# --- Step 4: Validate shape and fix if needed ---
if len(feature_vector) == 0:
    raise ValueError("❌ Feature vector is empty. Check if features were extracted properly.")

# Some rows might have inconsistent feature lengths; drop or fix them
max_len = max(len(f) for f in feature_vector if isinstance(f, (list, np.ndarray)))
feature_vector = np.array([np.pad(f, (0, max_len - len(f)), 'constant') if len(f) < max_len else f for f in feature_vector])

# --- Step 5: Ensure 2D shape ---
if feature_vector.ndim == 1:
    feature_vector = feature_vector.reshape(-1, 1)

print("✅ Feature vector shape:", feature_vector.shape)

# --- Step 6: Apply StandardScaler ---
sc = StandardScaler()
feature_vector_scaled = sc.fit_transform(feature_vector)

print("✅ Feature scaling complete. Scaled shape:", feature_vector_scaled.shape)

# --- Optional: Normalize additional metadata columns (if any) ---
# Example:
# metadata_cols = ['Age', 'PSA', 'Prostate_Volume']
# metadata_scaled = sc.fit_transform(metadata_final[metadata_cols])



# %% Cell 50
# Extract feature_vector and normalize it
feature_vector = np.array(metadata_final['Concatenated_Features'].tolist())
sc = StandardScaler()
feature_vector_scaled = sc.fit_transform(feature_vector)

# Extract additional metadata columns and normalize them
metadata_columns = ['psa', 'psad', 'prostate_volume','num_lesions','max_prim_score','max_sec_score','max_gleason_scores']
metadata_values = metadata_final[metadata_columns].values  
# scaler_metadata = StandardScaler()
metadata_scaled = sc.fit_transform(metadata_values)

# Concatenate metadata with feature_vector
x = np.hstack((metadata_scaled, feature_vector_scaled)) 

# Target variable
y = metadata_final[['case_ISUP','case_csPCa']].values  
le = LabelEncoder()
 

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, shuffle=True, random_state=42, stratify=y)


# Extract ISUP grades & corresponding clinical significance
case_ISUP = y_test[:,0]
case_csPCa = y_test[:,1]

# Consider only clinical significance for classification
y_train = y_train[:,1]
y_test = y_test[:,1]

# Perform label encoding of the target variable
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Define the model
with_metadata_model = Sequential([
    Input(shape=(x.shape[1],)),  
    BatchNormalization(),
    
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')  
])

# Compile the model
with_metadata_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)

# Callbacks: EarlyStopping & ReduceLROnPlateau
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# Train the model
history = with_metadata_model.fit(
    x_train, y_train, validation_split=0.30, epochs=30, batch_size=32, callbacks=callbacks, verbose=1
)

# Evaluating the model on test data
test_loss, test_acc, test_auc = with_metadata_model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc:.4f} Test AUC: {test_auc:.4f}')


# %% Cell 51
y_probs = with_metadata_model.predict(x_test)
y_pred = (y_probs >= 0.5).astype(int)
print("Classification Report:\n", classification_report(y_test, y_pred))


# %% Cell 52
# To compute grade-wise performance
results_df = pd.DataFrame()
results_df['case_ISUP'] = case_ISUP
results_df['case_csPCa'] = case_csPCa
results_df['predicted_csPCa'] = y_pred

results_df['TP'] = (results_df['case_csPCa'] == 1) & (results_df['predicted_csPCa'] == 1)
results_df['FN'] = (results_df['case_csPCa'] == 1) & (results_df['predicted_csPCa'] == 0)
results_df['FP'] = (results_df['case_csPCa'] == 0) & (results_df['predicted_csPCa'] == 1)
results_df['TN'] = (results_df['case_csPCa'] == 0) & (results_df['predicted_csPCa'] == 0)

# Convert boolean to integers
results_df[['TP', 'FN', 'FP', 'TN']] = results_df[['TP', 'FN', 'FP', 'TN']].astype(int)

def calculate_precision_recall(df):
    grades = sorted(df['case_ISUP'].unique())  # Unique ISUP grades
    result_list = []

    for grade in grades:
        grade_df = df[df['case_ISUP'] == grade]  # Filter by grade

        tp = grade_df['TP'].sum()
        fn = grade_df['FN'].sum()
        fp = df[df['case_ISUP'] != grade]['FP'].sum()  # FP should be outside the grade
        tn = grade_df['TN'].sum()

        if grade >= 2:
            # Use original TP-based formula for grades 2 and above
            precision = round(tp / (tp + fp), 2) if (tp + fp) > 0 else 0
            recall = round(tp / (tp + fn), 2) if (tp + fn) > 0 else 0
        else:
            # Use TN-based formula for grades 0 and 1
            precision = round(tn / (tn + fn), 2) if (tn + fn) > 0 else 0
            recall = round(tn / (tn + fp), 2) if (tn + fp) > 0 else 0

        result_list.append((grade, precision, recall))

    return pd.DataFrame(result_list, columns=['Grade', 'Precision', 'Recall'])

# Compute precision and recall
metrics_df = calculate_precision_recall(results_df)
metrics_df


# %% Cell 53
y_pred = (y_probs >= 0.5).astype(int)

TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

PPR = TP / (TP + FP)  # Positive Prediction Rate
NPR = TN / (TN + FN)  # Negative Prediction Rate

print(f"Positive Prediction Rate (PPR): {PPR:.3f}")
print(f"Negative Prediction Rate (NPR): {NPR:.3f}")


# %% Cell 54
def calculate_total_costs(y_true, y_probs, fp_cost=None, fn_cost=None):
    fp_cost = fp_cost if fp_cost is not None else 25000
    fn_cost = fn_cost if fn_cost is not None else 500000
    
    # TP and TN costs are 25% of FN and FP respectively
    tp_cost = 0.25 * fn_cost
    tn_cost = 0.25 * fp_cost

    best_threshold = None
    best_total_cost = float('inf')

    thresholds = np.arange(0.25, 0.65, 0.01)
    cost_data = []

    prev_total_cost = None

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        total_cost = (fp * fp_cost) + (fn * fn_cost) - (tp * tp_cost) - (tn * tn_cost)  
        formatted_cost = f"₹{total_cost:,}"
        
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_threshold = threshold
            
        if total_cost != prev_total_cost:
            cost_data.append([threshold, tp, tn, fp, fn, formatted_cost])
            prev_total_cost = total_cost

    cost_df = pd.DataFrame(cost_data, columns=['Threshold', 'TP', 'TN', 'FP', 'FN', 'Total Cost'])
    
    return cost_df, best_threshold, best_total_cost


fp_cost = int(input("Enter cost for False Positive(taking MRI scans): ") or 25000)
fn_cost = int(input("Enter cost for False Negative(surgery,chemotherapy etc): ") or 500000)

cost_df,best_threshold,best_total_cost = calculate_total_costs(y_test, y_probs, fp_cost=fp_cost, fn_cost=fn_cost)


# %% Cell 55
cost_df[:]


# %% Cell 56
print("\nBest Threshold:", round(best_threshold,2))
print(f"Minimum Net Cost: ₹{best_total_cost:,}")


# %% Cell 57
# Get y_pred for the best threshold
y_pred_best = (y_probs >= best_threshold).astype(int)

TN, FP, FN, TP = confusion_matrix(y_test, y_pred_best).ravel()

PPR = TP / (TP + FP)  # Positive Prediction Rate
NPR = TN / (TN + FN)  # Negative Prediction Rate

print(f"Positive Prediction Rate (PPR): {PPR:.3f}")
print(f"Negative Prediction Rate (NPR): {NPR:.3f}")
print("\nClassification Report based on the best Threshold:\n",classification_report(y_test, y_pred_best))


# %% Cell 58
# To compute grade-wise performance
results_df_best = pd.DataFrame()
results_df_best['case_ISUP_best'] = case_ISUP
results_df_best['case_csPCa_best'] = case_csPCa
results_df_best['predicted_csPCa_best'] = y_pred_best

results_df_best['TP_best'] = (results_df_best['case_csPCa_best'] == 1) & (results_df_best['predicted_csPCa_best'] == 1)
results_df_best['FN_best'] = (results_df_best['case_csPCa_best'] == 1) & (results_df_best['predicted_csPCa_best'] == 0)
results_df_best['FP_best'] = (results_df_best['case_csPCa_best'] == 0) & (results_df_best['predicted_csPCa_best'] == 1)
results_df_best['TN_best'] = (results_df_best['case_csPCa_best'] == 0) & (results_df_best['predicted_csPCa_best'] == 0)

# Convert boolean to integers
results_df_best[['TP_best', 'FN_best', 'FP_best', 'TN_best']] = results_df_best[['TP_best', 'FN_best', 'FP_best', 'TN_best']].astype(int)

def calculate_precision_recall_best(df):
    grades = sorted(df['case_ISUP_best'].unique())  # Unique ISUP grades
    result_list = []

    for grade in grades:
        grade_df = df[df['case_ISUP_best'] == grade]  # Filter by grade

        tp_best = grade_df['TP_best'].sum()
        fn_best = grade_df['FN_best'].sum()
        fp_best = df[df['case_ISUP_best'] != grade]['FP_best'].sum()  # FP should be outside the grade
        tn_best = grade_df['TN_best'].sum()

        if grade >= 2:
            # Use original TP-based formula for grades 2 and above
            precision_best = round(tp_best / (tp_best + fp_best), 2) if (tp_best + fp_best) > 0 else 0
            recall_best = round(tp_best / (tp_best + fn_best), 2) if (tp_best + fn_best) > 0 else 0
        else:
            # Use TN-based formula for grades 0 and 1
            precision_best = round(tn_best / (tn_best + fn_best), 2) if (tn_best + fn_best) > 0 else 0
            recall_best = round(tn_best / (tn_best + fp_best), 2) if (tn_best + fp_best) > 0 else 0

        result_list.append((grade, precision_best, recall_best))

    return pd.DataFrame(result_list, columns=['Grade', 'Precision_best', 'Recall_best'])

# Compute precision and recall
metrics_df_best = calculate_precision_recall_best(results_df_best)
metrics_df_best



# %% Cell 59
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, y_pred)  # Get false positive & true positive rates
roc_auc = auc(fpr, tpr)  # Compute AUC

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (with metadata)')
plt.legend(loc="lower right")
plt.grid()
plt.show()


# %% [markdown]
# ### Model building (without including metadata for classification)


# %% Cell 61
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Extract feature_vector and normalize it
x_without = np.array(metadata_final['Concatenated_Features'].tolist())
sc = StandardScaler()
x_without = sc.fit_transform(x_without)  # Normalize feature vectors

# Target variable
y_without = metadata_final[['case_ISUP', 'case_csPCa']].values  
le = LabelEncoder()

# Train-test split
x_train_without, x_test_without, y_train_without, y_test_without = train_test_split(
    x_without, y_without, test_size=0.05, shuffle=True, random_state=42, stratify=y_without)

# Extract ISUP grades & corresponding clinical significance
case_ISUP_without = y_test_without[:, 0]
case_csPCa_without = y_test_without[:, 1]

# Consider only clinical significance for classification
y_train_without = y_train_without[:, 1]
y_test_without = y_test_without[:, 1]

# Perform label encoding of the target variable
y_train_without = le.fit_transform(y_train_without)
y_test_without = le.transform(y_test_without)

# Define the model
without_metadata_model = Sequential([
    Input(shape=(x_without.shape[1],)),  # Dynamically set input shape
    BatchNormalization(),
    
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')  # Output layer
])

# Compile the model
without_metadata_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# Train the model
history = without_metadata_model.fit(
    x_train_without, y_train_without, validation_split=0.30, epochs=30, batch_size=32, callbacks=callbacks, verbose=1
)

# Evaluating the model on test data
test_loss_without, test_acc_without, test_auc_without = without_metadata_model.evaluate(x_test_without, y_test_without)
print(f'\nTest accuracy: {test_acc_without:.4f} Test AUC: {test_auc_without:.4f}')



# %% Cell 62
y_prob_without = without_metadata_model.predict(x_test_without)
y_pred_without = (y_prob_without > 0.5).astype(int)  # Predictions

# Performance evaluation
print("AUC Score: ", round(roc_auc_score(y_test_without, y_prob_without), 3))
print("Classification Report:\n", classification_report(y_test_without, y_pred_without))


# %% Cell 63
# Creating results DataFrame
results_df_without = pd.DataFrame()
results_df_without['case_ISUP'] = case_ISUP_without
results_df_without['case_csPCa'] = case_csPCa_without
results_df_without['predicted_csPCa'] = y_pred_without

# Efficient vectorized calculations for TP, FN, FP, TN
results_df_without['TP'] = (results_df_without['case_csPCa'] == 1) & (results_df_without['predicted_csPCa'] == 1)
results_df_without['FN'] = (results_df_without['case_csPCa'] == 1) & (results_df_without['predicted_csPCa'] == 0)
results_df_without['FP'] = (results_df_without['case_csPCa'] == 0) & (results_df_without['predicted_csPCa'] == 1)
results_df_without['TN'] = (results_df_without['case_csPCa'] == 0) & (results_df_without['predicted_csPCa'] == 0)

# Convert boolean to integers
results_df_without[['TP', 'FN', 'FP', 'TN']] = results_df_without[['TP', 'FN', 'FP', 'TN']].astype(int)

def calculate_precision_recall(df):
    grades = sorted(df['case_ISUP'].unique())  # Unique ISUP grades
    result_list = []

    for grade in grades:
        grade_df = df[df['case_ISUP'] == grade]  # Filter by grade

        tp = grade_df['TP'].sum()
        fn = grade_df['FN'].sum()
        fp = df[df['case_ISUP'] != grade]['FP'].sum()  # FP should be outside the grade
        tn = grade_df['TN'].sum()

        if grade >= 2:
            # Use original TP-based formula for grades 2 and above
            precision = round(tp / (tp + fp), 2) if (tp + fp) > 0 else 0
            recall = round(tp / (tp + fn), 2) if (tp + fn) > 0 else 0
        else:
            # Use TN-based formula for grades 0 and 1
            precision = round(tn / (tn + fn), 2) if (tn + fn) > 0 else 0
            recall = round(tn / (tn + fp), 2) if (tn + fp) > 0 else 0

        result_list.append((grade, precision, recall))

    return pd.DataFrame(result_list, columns=['Grade', 'Precision', 'Recall'])

# Compute precision and recall
metrics_df_without = calculate_precision_recall(results_df_without)
print(metrics_df_without)


# %% Cell 64
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr_without, tpr_without, _without = roc_curve(y_test_without, y_pred_without)  # Get false positive & true positive rates
roc_auc_without = auc(fpr_without, tpr_without)  # Compute AUC

plt.figure(figsize=(8, 6))
plt.plot(fpr_without, tpr_without, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_without:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve(without metadata)')
plt.legend(loc="lower right")
plt.grid()
plt.show()


# %% Cell 65



# %% Cell 66



# %% Cell 67

