import { useEffect } from 'react';

export const useCornerstone = (elementRef) => {
  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const cornerstone = window.cornerstone;
    const cornerstoneTools = window.cornerstoneTools;
    const cornerstoneWADOImageLoader = window.cornerstoneWADOImageLoader;
    const dicomParser = window.dicomParser;

    if (!cornerstone || !cornerstoneTools || !cornerstoneWADOImageLoader || !dicomParser) {
      console.error("Cornerstone JS libraries not loaded in window scope.");
      return;
    }

    // Initialize dependencies on the WADO Loader
    cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
    cornerstoneWADOImageLoader.external.dicomParser = dicomParser;
    
    // Disable web workers to prevent Vite bundling import errors
    try {
      cornerstoneWADOImageLoader.configure({ useWebWorkers: false });
    } catch (e) {
      // Already configured
    }

    // Enable element
    try {
      cornerstone.enable(element);
    } catch (e) {
      // Already enabled
    }

    // Initialize tools
    try {
      cornerstoneTools.init();
    } catch (e) {
      // Already initialized
    }

    // Register basic tools
    try {
      cornerstoneTools.addTool(cornerstoneTools.WwwcTool);
      cornerstoneTools.addTool(cornerstoneTools.PanTool);
      cornerstoneTools.addTool(cornerstoneTools.ZoomTool);
    } catch (e) {
      // Already registered
    }

    // Set Window/Level active as default tool (bound to left-mouse button)
    try {
      cornerstoneTools.setToolActive('Wwwc', { mouseButtonMask: 1 });
    } catch (e) {
      console.error("Failed to activate default tool:", e);
    }

    return () => {
      // Cleanup to prevent memory leaks on navigate
      try {
        cornerstone.disable(element);
      } catch (e) {
        // Element was already disabled or not initialized
      }
    };
  }, [elementRef]);
};
