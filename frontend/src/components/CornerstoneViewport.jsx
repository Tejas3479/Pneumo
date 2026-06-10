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
      
      if (onImageLoaded) {
        onImageLoaded({ width: image.width, height: image.height });
      }
      
      // Set active tool initially
      const cornerstoneTools = window.cornerstoneTools;
      if (cornerstoneTools) {
        cornerstoneTools.setToolActive(activeTool, { mouseButtonMask: 1 });
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
    cornerstoneTools.setToolPassive('Wwwc');
    cornerstoneTools.setToolPassive('Pan');
    cornerstoneTools.setToolPassive('Zoom');
    
    // Activate current
    cornerstoneTools.setToolActive(activeTool, { mouseButtonMask: 1 });
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
      <div ref={dicomRef} className="w-full h-full absolute top-0 left-0"></div>
      
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
