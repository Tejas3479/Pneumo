import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, AlertCircle, FileX } from 'lucide-react';

const MAX_FILE_SIZE = 52428800; // 50 MB in bytes

export default function UploadZone({ onFileSelected, isLoading }) {
  const [rejectionReason, setRejectionReason] = useState('');

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    setRejectionReason('');

    if (rejectedFiles && rejectedFiles.length > 0) {
      const firstRejection = rejectedFiles[0];
      const errors = firstRejection.errors.map(e => e.message).join(', ');
      setRejectionReason(errors);
      return;
    }

    if (acceptedFiles && acceptedFiles.length > 0 && !isLoading) {
      onFileSelected(acceptedFiles[0]);
    }
  }, [onFileSelected, isLoading]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      // Standard web image types
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      // DICOM: use octet-stream with .dcm extension — browsers don't recognize application/dicom
      'application/octet-stream': ['.dcm'],
    },
    disabled: isLoading,
    multiple: false,
    maxSize: MAX_FILE_SIZE,
    // Custom validator to also allow .dcm files regardless of MIME type
    validator: (file) => {
      const name = file.name.toLowerCase();
      if (name.endsWith('.dcm')) return null; // Allow all .dcm files
      if (file.type.startsWith('image/')) return null;
      return {
        code: 'file-invalid-type',
        message: 'Only DICOM (.dcm), PNG, and JPEG files are accepted.',
      };
    },
  });

  const dropZoneClass = isDragReject
    ? 'border-brand-pathology bg-brand-pathology/5 shadow-lg shadow-brand-pathology/10'
    : isDragActive
      ? 'border-brand-violet bg-brand-violet/5 shadow-lg shadow-brand-violet/10 scale-[0.99]'
      : 'border-slate-800 bg-slate-950/40 hover:border-brand-cyan/40 hover:bg-slate-900/10';

  return (
    <div className="space-y-3">
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-300 min-h-[280px] flex flex-col items-center justify-center gap-4
          ${dropZoneClass}
          ${isLoading ? 'opacity-50 cursor-not-allowed pointer-events-none' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <div className={`p-5 rounded-2xl bg-slate-900/60 border border-brand-border transition-all duration-300 ${isDragActive ? (isDragReject ? 'text-brand-pathology border-brand-pathology/30' : 'text-brand-violet -translate-y-1') : 'text-brand-cyan'}`}>
          {isDragReject ? (
            <FileX className="w-10 h-10" />
          ) : (
            <UploadCloud className="w-10 h-10" />
          )}
        </div>

        <div className="space-y-1">
          <h3 className="font-heading font-semibold text-base text-slate-100">
            {isDragReject
              ? 'Unsupported File Type'
              : isDragActive
                ? 'Drop the chest radiograph here'
                : 'Acquire Chest X-Ray'}
          </h3>
          <p className="text-xs text-brand-textMuted max-w-xs mx-auto leading-relaxed">
            {isDragReject
              ? 'Only DICOM (.dcm), PNG, and JPEG files are accepted.'
              : 'Drag & drop a clinical DICOM (.dcm), PNG, or JPEG image, or click to browse files'}
          </p>
          <p className="text-[10px] text-brand-textMuted/60 mt-1">
            Max file size: 50 MB
          </p>
        </div>

        {/* Clickable browse — stopPropagation so dropzone doesn't open dialog twice */}
        <button
          type="button"
          className="mt-2 text-xs font-semibold px-4 py-2 rounded-lg bg-brand-cyan/10 hover:bg-brand-cyan/20 border border-brand-cyan/35 text-brand-cyan transition-all duration-300 uppercase tracking-wider disabled:opacity-50"
          disabled={isLoading}
          onClick={(e) => e.stopPropagation()}
        >
          Browse Files
        </button>
      </div>

      {/* Rejection error message */}
      {rejectionReason && (
        <div className="flex items-start gap-2 px-4 py-3 rounded-xl border border-brand-pathology/30 bg-brand-pathology/5 text-brand-pathology text-xs">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold block">File rejected:</span>
            <span>{rejectionReason}</span>
          </div>
        </div>
      )}
    </div>
  );
}
