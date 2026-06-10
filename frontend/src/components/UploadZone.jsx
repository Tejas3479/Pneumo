import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud } from 'lucide-react';

export default function UploadZone({ onFileSelected, isLoading }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0 && !isLoading) {
      onFileSelected(acceptedFiles[0]);
    }
  }, [onFileSelected, isLoading]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg'],
      'application/dicom': ['.dcm'],
    },
    disabled: isLoading,
    multiple: false,
  });

  return (
    <div 
      {...getRootProps()} 
      className={`
        border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-300 min-h-[300px] flex flex-col items-center justify-center gap-4
        ${isDragActive 
          ? 'border-brand-violet bg-brand-violet/5 shadow-lg shadow-brand-violet/10 scale-[0.99]' 
          : 'border-slate-800 bg-slate-950/40 hover:border-brand-cyan/40 hover:bg-slate-900/10'}
        ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      <input {...getInputProps()} />
      
      <div className={`p-5 rounded-2xl bg-slate-900/60 border border-brand-border text-brand-cyan transition-transform duration-300 ${isDragActive ? 'text-brand-violet -translate-y-1' : ''}`}>
        <UploadCloud className="w-10 h-10" />
      </div>

      <div className="space-y-1">
        <h3 className="font-heading font-semibold text-base text-slate-100">
          {isDragActive ? 'Drop the chest radiograph here' : 'Acquire Chest X-Ray'}
        </h3>
        <p className="text-xs text-brand-textMuted max-w-xs mx-auto leading-relaxed">
          Drag & drop a clinical DICOM (.dcm), PNG, or JPEG image, or click to browse files
        </p>
      </div>

      <button 
        type="button" 
        className="mt-2 text-xs font-semibold px-4 py-2 rounded-lg bg-brand-cyan/10 hover:bg-brand-cyan/20 border border-brand-cyan/35 text-brand-cyan transition-all duration-300 uppercase tracking-wider"
        disabled={isLoading}
      >
        Browse Files
      </button>
    </div>
  );
}
