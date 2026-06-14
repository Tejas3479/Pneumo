import React, { useEffect, useRef, useState } from 'react';
import { useCornerstone } from '../hooks/useCornerstone';

export default function CornerstoneViewport({ 
  imageId, 
  overlayUrl, 
  overlayEnabled, 
  activeTool,
  onImageLoaded 
}) {
  const dicomRef = useRef(null);
  const overlayRef = useRef(null);
  const [originalDimensions, setOriginalDimensions] = useState({ width: 0, height: 0 });
  const [metadata, setMetadata] = useState(null);

  // Hook handles cornerstone enable/disable on mounting dicomRef
  useCornerstone(dicomRef);

  // Load new image whenever imageId changes
  useEffect(() => {
    const element = dicomRef.current;
    if (!element || !imageId) return;

    const cornerstone = window.cornerstone;
    
    cornerstone.loadImage(imageId).then((image) => {
      cornerstone.displayImage(element, image);
      setOriginalDimensions({ width: image.width, height: image.height });
      
      if (image && image.data) {
        try {
          const dataset = image.data;
          const patientNameObj = dataset.string('x00100010') || 'N/A';
          const patientId = dataset.string('x00100020') || 'N/A';
          const patientSex = dataset.string('x00100040') || 'N/A';
          const patientAge = dataset.string('x00100030') || 'N/A';
          const studyDate = dataset.string('x00080020') || 'N/A';
          const studyDesc = dataset.string('x00081030') || 'No Description';
          const modality = dataset.string('x00080060') || 'DX';
          
          setMetadata({
            patientName: typeof patientNameObj === 'object' ? (patientNameObj.Alphabetic || 'N/A') : patientNameObj,
            patientId,
            patientSex,
            patientAge,
            studyDate,
            studyDesc,
            modality
          });
        } catch (e) {
          console.warn("Failed to extract metadata from DICOM WADO image dataset:", e);
        }
      }
      
      if (onImageLoaded) {
        onImageLoaded({ width: image.width, height: image.height });
      }
      
      // Set active tool initially
      const cornerstoneTools = window.cornerstoneTools;
      if (cornerstoneTools) {
        try {
          cornerstoneTools.setToolActive(activeTool, { mouseButtonMask: 1 });
        } catch (e) {}
      }
    }).catch((err) => {
      console.error("Cornerstone image loading error:", err);
    });
  }, [imageId, activeTool, onImageLoaded]);

  // Synchronize active tool changes
  useEffect(() => {
    const cornerstoneTools = window.cornerstoneTools;
    if (!cornerstoneTools || !activeTool) return;
    
    // Deactivate all first
    const tools = ['Wwwc', 'Pan', 'Zoom', 'Length', 'Angle', 'Probe', 'RectangleRoi', 'EllipticalRoi'];
    tools.forEach(t => {
      try {
        cornerstoneTools.setToolPassive(t);
      } catch (e) {}
    });
    
    // Activate current
    try {
      cornerstoneTools.setToolActive(activeTool, { mouseButtonMask: 1 });
    } catch (e) {
      console.error("Failed to activate tool:", activeTool, e);
    }
  }, [activeTool]);

  // Synchronize overlay position on rendering events
  useEffect(() => {
    const element = dicomRef.current;
    if (!element) return;

    const syncOverlay = (e) => {
      const overlay = overlayRef.current;
      if (!overlay || !overlayEnabled || originalDimensions.width === 0) return;

      const eventDetail = e.detail;
      const canvas = eventDetail.canvasContext.canvas;
      const viewport = eventDetail.viewport;
      
      const scale = viewport.scale;
      const translation = viewport.translation;
      
      const displayWidth = originalDimensions.width * scale;
      const displayHeight = originalDimensions.height * scale;
      
      // Calculate top/left offset relative to container boundary (translation in screen pixels)
      const offsetX = canvas.width / 2 + translation.x - displayWidth / 2;
      const offsetY = canvas.height / 2 + translation.y - displayHeight / 2;
      
      overlay.style.width = displayWidth + 'px';
      overlay.style.height = displayHeight + 'px';
      overlay.style.left = offsetX + 'px';
      overlay.style.top = offsetY + 'px';
    };

    element.addEventListener('cornerstoneimagerendered', syncOverlay);

    // Run manual sync initially in case image is rendered already
    const cornerstone = window.cornerstone;
    if (cornerstone && overlayEnabled) {
      try {
        const viewport = cornerstone.getViewport(element);
        const enabledElement = cornerstone.getEnabledElement(element);
        if (viewport && enabledElement && enabledElement.image) {
          syncOverlay({
            detail: {
              canvasContext: enabledElement.canvas.getContext('2d'),
              viewport: viewport
            }
          });
        }
      } catch (err) {
        // Ignored
      }
    }

    return () => {
      element.removeEventListener('cornerstoneimagerendered', syncOverlay);
    };
  }, [overlayEnabled, originalDimensions]);

  return (
    <div className="relative w-full h-full min-h-[400px] bg-black rounded-xl overflow-hidden border border-brand-border">
      {/* Cornerstone Render Target */}
      <div ref={dicomRef} id="dicomViewport" className="w-full h-full absolute top-0 left-0"></div>
      
      {/* Patient Tags Overlay */}
      {metadata && (
        <div className="absolute top-4 left-4 z-20 pointer-events-none select-none bg-slate-950/80 border border-brand-border/40 backdrop-blur-sm p-3 rounded-lg text-[10px] text-slate-300 space-y-1 font-mono">
          <div><span className="text-brand-cyan font-bold uppercase">Patient:</span> {metadata.patientName}</div>
          <div><span className="text-brand-cyan font-bold uppercase">ID:</span> {metadata.patientId}</div>
          <div><span className="text-brand-cyan font-bold uppercase">Age/Sex:</span> {metadata.patientAge} / {metadata.patientSex}</div>
          <div><span className="text-brand-cyan font-bold uppercase">Study Date:</span> {metadata.studyDate}</div>
          <div><span className="text-brand-cyan font-bold uppercase">Desc:</span> {metadata.studyDesc}</div>
          <div><span className="text-brand-cyan font-bold uppercase">Modality:</span> {metadata.modality}</div>
        </div>
      )}
      
      {/* Absolute Heatmap Overlay Image */}
      {overlayUrl && (
        <img 
          ref={overlayRef} 
          src={overlayUrl} 
          alt="Heatmap overlay" 
          className="absolute pointer-events-none select-none opacity-60 z-10"
          style={{ display: overlayEnabled ? 'block' : 'none' }}
        />
      )}
    </div>
  );
}
