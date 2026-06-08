import os
import numpy as np
import cv2
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, ImplicitVRLittleEndian

def create_secondary_capture(original_ds, heatmap_np, output_path):
    """
    Creates a Secondary Capture DICOM image blending the original image and Grad-CAM heatmap.
    """
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.7')  # Secondary Capture Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    sc_ds = Dataset()
    sc_ds.file_meta = file_meta
    sc_ds.is_little_endian = True
    sc_ds.is_implicit_VR = False
    
    # Copy Patient and Study tags from original
    patient_tags = ['PatientName', 'PatientID', 'PatientSex', 'PatientBirthDate']
    for tag in patient_tags:
        if hasattr(original_ds, tag):
            setattr(sc_ds, tag, getattr(original_ds, tag))
            
    study_tags = ['StudyInstanceUID', 'StudyDate', 'StudyTime', 'AccessionNumber', 'ReferringPhysicianName']
    for tag in study_tags:
        if hasattr(original_ds, tag):
            setattr(sc_ds, tag, getattr(original_ds, tag))
            
    # New Series and SOP Instance UID for the SC image
    sc_ds.SeriesInstanceUID = generate_uid()
    sc_ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    sc_ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    sc_ds.SeriesNumber = "999"
    sc_ds.InstanceNumber = "1"
    
    # RGB image attributes
    sc_ds.SamplesPerPixel = 3
    sc_ds.PhotometricInterpretation = "RGB"
    sc_ds.PlanarConfiguration = 0
    sc_ds.Rows = 224
    sc_ds.Columns = 224
    sc_ds.BitsAllocated = 8
    sc_ds.BitsStored = 8
    sc_ds.HighBit = 7
    sc_ds.PixelRepresentation = 0
    
    # Extract original pixel data, normalize to [0, 1]
    try:
        pixel_array = original_ds.pixel_array.astype(np.float32)
        min_val = pixel_array.min()
        max_val = pixel_array.max()
        if max_val - min_val > 0:
            pixel_array = (pixel_array - min_val) / (max_val - min_val)
        else:
            pixel_array = np.zeros_like(pixel_array)
    except Exception:
        pixel_array = np.zeros((224, 224), dtype=np.float32)
        
    pixel_array_uint8 = (pixel_array * 255.0).astype(np.uint8)
    
    # Resize original image to 224x224 and convert to 3-channel RGB
    orig_resized = cv2.resize(pixel_array_uint8, (224, 224))
    rgb_orig = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2RGB)
    
    # Resize and color map the Grad-CAM heatmap
    heatmap_uint8 = np.uint8(255 * heatmap_np)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend original and heatmap 50/50
    blended = cv2.addWeighted(rgb_orig, 0.5, heatmap_rgb, 0.5, 0)
    
    sc_ds.PixelData = blended.tobytes()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sc_ds.save_as(output_path, write_like_original=False)
    return sc_ds

def create_dicom_sr(original_ds, prediction_prob, prediction_label, sc_sop_instance_uid=None, sc_series_instance_uid=None):
    """
    Creates a Basic Text Structured Report (SR) file.
    """
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.88.11')  # Basic Text SR Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    
    sr_ds = Dataset()
    sr_ds.file_meta = file_meta
    sr_ds.is_little_endian = True
    sr_ds.is_implicit_VR = True
    
    # Copy patient / study tags
    patient_tags = ['PatientName', 'PatientID', 'PatientSex', 'PatientBirthDate']
    for tag in patient_tags:
        if hasattr(original_ds, tag):
            setattr(sr_ds, tag, getattr(original_ds, tag))
            
    study_tags = ['StudyInstanceUID', 'StudyDate', 'StudyTime', 'AccessionNumber', 'ReferringPhysicianName']
    for tag in study_tags:
        if hasattr(original_ds, tag):
            setattr(sr_ds, tag, getattr(original_ds, tag))
            
    # Series / Instance level tags
    sr_ds.SeriesInstanceUID = generate_uid()
    sr_ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    sr_ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    sr_ds.SeriesNumber = "990"
    sr_ds.InstanceNumber = "1"
    
    sr_ds.ValueType = "CONTAINER"
    
    # Concept Name Code Sequence: "Radiology Report"
    concept_name = Dataset()
    concept_name.CodeValue = "11528-7"
    concept_name.CodingSchemeDesignator = "LN"
    concept_name.CodeMeaning = "Radiology Report"
    sr_ds.ConceptNameCodeSequence = Sequence([concept_name])
    
    sr_ds.ContinuityOfContent = "SEPARATE"
    sr_ds.CompletionFlag = "COMPLETE"
    sr_ds.VerificationFlag = "UNVERIFIED"
    
    # Content Template Sequence
    template_ds = Dataset()
    template_ds.MappingResource = "DCMR"
    template_ds.TemplateIdentifier = "2000"
    sr_ds.ContentTemplateSequence = Sequence([template_ds])
    
    content_sequence = []
    
    # Item 1: Finding Category (Code)
    item_finding = Dataset()
    item_finding.RelationshipType = "CONTAINS"
    item_finding.ValueType = "CODE"
    concept_name_finding = Dataset()
    concept_name_finding.CodeValue = "121070"
    concept_name_finding.CodingSchemeDesignator = "DCM"
    concept_name_finding.CodeMeaning = "Findings"
    item_finding.ConceptNameCodeSequence = Sequence([concept_name_finding])
    
    concept_finding_val = Dataset()
    concept_finding_val.CodeValue = "PNEUMOTHORAX_DETECT"
    concept_finding_val.CodingSchemeDesignator = "PNEUMODETECT"
    concept_finding_val.CodeMeaning = f"Pneumothorax AI Classification: {prediction_label}"
    item_finding.ConceptCodeSequence = Sequence([concept_finding_val])
    content_sequence.append(item_finding)
    
    # Item 2: Finding Discussion (Text)
    item_prob = Dataset()
    item_prob.RelationshipType = "CONTAINS"
    item_prob.ValueType = "TEXT"
    concept_name_prob = Dataset()
    concept_name_prob.CodeValue = "121071"
    concept_name_prob.CodingSchemeDesignator = "DCM"
    concept_name_prob.CodeMeaning = "Finding Discussion"
    item_prob.ConceptNameCodeSequence = Sequence([concept_name_prob])
    item_prob.TextValue = f"AI Probability score of Pneumothorax: {prediction_prob:.4f}. Classification Result: {prediction_label}."
    content_sequence.append(item_prob)
    
    # Item 3: Referenced Source Image
    item_ref_orig = Dataset()
    item_ref_orig.RelationshipType = "CONTAINS"
    item_ref_orig.ValueType = "IMAGE"
    concept_name_ref = Dataset()
    concept_name_ref.CodeValue = "121112"
    concept_name_ref.CodingSchemeDesignator = "DCM"
    concept_name_ref.CodeMeaning = "Source Image"
    item_ref_orig.ConceptNameCodeSequence = Sequence([concept_name_ref])
    
    ref_image_seq = Dataset()
    ref_image_seq.ReferencedSOPClassUID = original_ds.SOPClassUID
    ref_image_seq.ReferencedSOPInstanceUID = original_ds.SOPInstanceUID
    item_ref_orig.ReferencedSOPSequence = Sequence([ref_image_seq])
    content_sequence.append(item_ref_orig)
    
    # Item 4: Referenced Secondary Capture overlay
    if sc_sop_instance_uid:
        item_ref_sc = Dataset()
        item_ref_sc.RelationshipType = "CONTAINS"
        item_ref_sc.ValueType = "IMAGE"
        concept_name_sc = Dataset()
        concept_name_sc.CodeValue = "121113"
        concept_name_sc.CodingSchemeDesignator = "DCM"
        concept_name_sc.CodeMeaning = "Secondary Capture Saliency Overlay"
        item_ref_sc.ConceptNameCodeSequence = Sequence([concept_name_sc])
        
        ref_sc_seq = Dataset()
        ref_sc_seq.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        ref_sc_seq.ReferencedSOPInstanceUID = sc_sop_instance_uid
        item_ref_sc.ReferencedSOPSequence = Sequence([ref_sc_seq])
        content_sequence.append(item_ref_sc)
        
    sr_ds.ContentSequence = Sequence(content_sequence)
    return sr_ds
