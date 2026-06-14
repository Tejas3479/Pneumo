import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { useApp } from '../context/AppContext';
import { useTaskPolling } from '../hooks/useTaskPolling';
import CornerstoneViewport from '../components/CornerstoneViewport';
import { 
  ArrowLeft, 
  Search, 
  Move, 
  ZoomIn, 
  RotateCcw, 
  Layers, 
  Play, 
  FileDown, 
  FileText,
  Ruler,
  Compass,
  Activity,
  Square,
  Circle
} from 'lucide-react';

export default function StudyViewerPage() {
  const { studyUid } = useParams();
  const navigate = useNavigate();
  const { addNotification, settings } = useApp();

  // Viewport states
  const [seriesList, setSeriesList] = useState([]);
  const [instancesList, setInstancesList] = useState([]);
  const [activeSeries, setActiveSeries] = useState('');
  const [activeInstance, setActiveInstance] = useState('');
  const [activeTool, setActiveTool] = useState('Wwwc'); // Wwwc, Pan, Zoom
  const [overlayEnabled, setOverlayEnabled] = useState(false);

  // Prediction states
  const [predictionData, setPredictionData] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionTaskId, setPredictionTaskId] = useState(null);

  const { status: pollStatus, result: pollResult, error: pollError } = useTaskPolling(predictionTaskId);

  // Fetch Series & Instances hierarchy
  const loadStudyHierarchy = useCallback(async () => {
    try {
      const series = await api.getStudySeries(studyUid);
      setSeriesList(series);
      
      if (series.length > 0) {
        // Optional chaining — DICOM QIDO-RS tags may be absent
        const firstSeriesUid = series[0]?.["0020000E"]?.["Value"]?.[0];
        if (firstSeriesUid) {
          setActiveSeries(firstSeriesUid);
          await loadSeriesInstances(firstSeriesUid);
        }
      }
    } catch (err) {
      console.error(err);
      addNotification('Failed to query series hierarchy: ' + err.message, 'error');
    }
  }, [studyUid, addNotification]);

  const loadSeriesInstances = async (seriesUid) => {
    try {
      const instances = await api.getSeriesInstances(studyUid, seriesUid);
      
      // Filter image-only instances (exclude Secondary Capture and Structured Report UIDs)
      const imageInstances = instances.filter((inst) => {
        const sopClass = inst["00080016"]?.["Value"]?.[0];
        return sopClass !== '1.2.840.10008.5.1.4.1.1.7' && sopClass !== '1.2.840.10008.5.1.4.1.1.88.11';
      });

      setInstancesList(imageInstances);

      if (imageInstances.length > 0) {
        // Optional chaining — SOP Instance UID tag may be absent in some viewers
        const firstSopUid = imageInstances[0]?.["00080018"]?.["Value"]?.[0];
        setActiveInstance(firstSopUid || '');
      } else {
        setActiveInstance('');
        addNotification('No diagnostic image instances found in this series.', 'info');
      }
    } catch (err) {
      console.error(err);
      addNotification('Failed to query instances hierarchy: ' + err.message, 'error');
    }
  };

  // Fetch cached prediction
  const checkPredictionStatus = useCallback(async () => {
    setPredictionLoading(true);
    try {
      const data = await api.getStudyPrediction(studyUid);
      if (data.status === 'completed') {
        setPredictionData(data);
      } else {
        setPredictionData(null);
      }
    } catch (err) {
      console.error(err);
      setPredictionData(null);
    } finally {
      setPredictionLoading(false);
    }
  }, [studyUid]);

  useEffect(() => {
    loadStudyHierarchy();
    checkPredictionStatus();
  }, [loadStudyHierarchy, checkPredictionStatus]);

  // Handle enqueuing prediction
  const handleRunPrediction = async () => {
    setPredictionTaskId(null);
    try {
      const data = await api.predictStudy(studyUid, settings.modelType);
      if (data.task_id) {
        setPredictionTaskId(data.task_id);
        addNotification('Prediction enqueued. Polling Celery tasks...', 'info');
      }
    } catch (err) {
      console.error(err);
      addNotification('Failed to enqueue AI prediction: ' + err.message, 'error');
    }
  };

  // Handle polling completion
  useEffect(() => {
    if (pollStatus === 'SUCCESS' && pollResult) {
      setPredictionData(pollResult);
      setPredictionTaskId(null);
      addNotification('AI Diagnostics prediction completed.', 'success');
    } else if (pollStatus === 'FAILED') {
      addNotification('AI Prediction enqueued task failed: ' + pollError, 'error');
      setPredictionTaskId(null);
    }
  }, [pollStatus, pollResult, pollError, addNotification]);

  const getBaseURL = () => {
    return settings.serverUrl ? settings.serverUrl.replace(/\/$/, '') : window.location.origin;
  };

  const formatUrl = (url) => {
    if (!url) return '';
    if (url.startsWith('http://') || url.startsWith('https://')) return url;
    const base = settings.serverUrl ? settings.serverUrl.replace(/\/$/, '') : '';
    return `${base}${url.startsWith('/') ? '' : '/'}${url}`;
  };

  // Construct WADO URLs
  const imageId = activeSeries && activeInstance
    ? `wadouri:${getBaseURL()}/dicomweb/studies/${studyUid}/series/${activeSeries}/instances/${activeInstance}`
    : '';

  const overlayUrl = activeSeries && activeInstance
    ? `${getBaseURL()}/dicomweb/studies/${studyUid}/series/${activeSeries}/instances/${activeInstance}/heatmap`
    : '';

  const isPredicting = pollStatus === 'PENDING';

  return (
    <div className="space-y-6 animate-fade-in relative z-0">
      
      {/* Navigation Subheader */}
      <div className="flex items-center justify-between">
        <button
          onClick={() => navigate('/studies')}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-brand-border hover:bg-slate-900 text-xs font-semibold text-slate-300 transition-colors uppercase tracking-wider"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Registry</span>
        </button>
      </div>

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-8">
        
        {/* Left Column: Viewport (Span 3) */}
        <div className="xl:col-span-3 flex flex-col gap-4">
          
          {/* Viewport Toolbar controls */}
          <div className="p-4 rounded-xl border border-brand-border bg-slate-950/60 backdrop-blur flex flex-wrap items-center gap-3">
            <button
              onClick={() => setActiveTool('Wwwc')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors border ${activeTool === 'Wwwc' ? 'bg-brand-cyan border-brand-cyan text-slate-955 font-bold' : 'border-brand-border hover:bg-slate-900 text-slate-300'}`}
            >
              <Search className="w-3.5 h-3.5" />
              <span>Window/Level</span>
            </button>
            <button
              onClick={() => setActiveTool('Pan')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors border ${activeTool === 'Pan' ? 'bg-brand-cyan border-brand-cyan text-slate-955 font-bold' : 'border-brand-border hover:bg-slate-900 text-slate-300'}`}
            >
              <Move className="w-3.5 h-3.5" />
              <span>Pan</span>
            </button>
            <button
              onClick={() => setActiveTool('Zoom')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors border ${activeTool === 'Zoom' ? 'bg-brand-cyan border-brand-cyan text-slate-955 font-bold' : 'border-brand-border hover:bg-slate-900 text-slate-300'}`}
            >
              <ZoomIn className="w-3.5 h-3.5" />
              <span>Zoom</span>
            </button>
            <button
              onClick={() => setActiveTool('Length')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors border ${activeTool === 'Length' ? 'bg-brand-cyan border-brand-cyan text-slate-955 font-bold' : 'border-brand-border hover:bg-slate-900 text-slate-300'}`}
            >
              <Ruler className="w-3.5 h-3.5" />
              <span>Ruler</span>
            </button>
            <button
              onClick={() => setActiveTool('Angle')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors border ${activeTool === 'Angle' ? 'bg-brand-cyan border-brand-cyan text-slate-955 font-bold' : 'border-brand-border hover:bg-slate-900 text-slate-300'}`}
            >
              <Compass className="w-3.5 h-3.5" />
              <span>Angle</span>
            </button>
            <button
              onClick={() => setActiveTool('Probe')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors border ${activeTool === 'Probe' ? 'bg-brand-cyan border-brand-cyan text-slate-955 font-bold' : 'border-brand-border hover:bg-slate-900 text-slate-300'}`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>Probe</span>
            </button>
            <button
              onClick={() => setActiveTool('RectangleRoi')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors border ${activeTool === 'RectangleRoi' ? 'bg-brand-cyan border-brand-cyan text-slate-955 font-bold' : 'border-brand-border hover:bg-slate-900 text-slate-300'}`}
            >
              <Square className="w-3.5 h-3.5" />
              <span>Rect</span>
            </button>
            <button
              onClick={() => setActiveTool('EllipticalRoi')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors border ${activeTool === 'EllipticalRoi' ? 'bg-brand-cyan border-brand-cyan text-slate-955 font-bold' : 'border-brand-border hover:bg-slate-900 text-slate-300'}`}
            >
              <Circle className="w-3.5 h-3.5" />
              <span>Ellipse</span>
            </button>
            
            <span className="w-[1px] h-6 bg-brand-border mx-2"></span>

            <button
              onClick={() => {
                const element = document.getElementById('dicomViewport');
                if (element && window.cornerstone) {
                  window.cornerstone.reset(element);
                }
              }}
              className="flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider border border-brand-border hover:bg-slate-900 text-slate-300 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset</span>
            </button>

            <button
              onClick={() => setOverlayEnabled(!overlayEnabled)}
              disabled={!predictionData}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors border ${!predictionData ? 'opacity-40 cursor-not-allowed border-brand-border text-slate-500' : overlayEnabled ? 'bg-brand-violet border-brand-violet text-white font-bold' : 'border-brand-violet/40 hover:bg-brand-violet/15 text-brand-violet'}`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Overlay Heatmap</span>
            </button>

            <a
              href={`${getBaseURL()}/static/ohif/index.html?studyInstanceUID=${studyUid}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors border border-brand-cyan/40 hover:bg-brand-cyan/15 text-brand-cyan"
            >
              <FileText className="w-3.5 h-3.5" />
              <span>Advanced Medical Viewer</span>
            </a>
            
            {/* Series selector if multiple */}
            {seriesList.length > 1 && (
              <div className="ml-auto flex items-center gap-2">
                <span className="text-[10px] uppercase font-bold text-brand-textMuted">Series:</span>
                <select
                  value={activeSeries}
                  onChange={(e) => {
                    setActiveSeries(e.target.value);
                    loadSeriesInstances(e.target.value);
                  }}
                  className="bg-slate-900 border border-brand-border text-xs px-2 py-1 rounded outline-none text-slate-300"
                >
                  {seriesList.map((s) => {
                    const sUid = s?.["0020000E"]?.["Value"]?.[0] || 'unknown';
                    // Tag 00201209 = Number of Series Related Instances (correct tag, not 00200013 = Instance Number)
                    const numInstances = s?.["00201209"]?.["Value"]?.[0] ?? s?.["00200013"]?.["Value"]?.[0] ?? '?';
                    return <option key={sUid} value={sUid}>Series {sUid.substring(Math.max(0, sUid.length - 6))} ({numInstances} frames)</option>
                  })}
                </select>
              </div>
            )}
          </div>

          {/* Cornerstone viewport renderer */}
          <div className="flex-1 min-h-[600px] h-[600px]">
            {imageId ? (
              <CornerstoneViewport
                imageId={imageId}
                overlayUrl={overlayUrl}
                overlayEnabled={overlayEnabled}
                activeTool={activeTool}
              />
            ) : (
              <div className="w-full h-full rounded-xl bg-black border border-brand-border flex items-center justify-center text-slate-500 text-xs">
                No active series/instance loaded.
              </div>
            )}
          </div>
        </div>

        {/* Right Column: AI Annotations (Span 1) */}
        <div className="space-y-6">
          <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4">
            <h3 className="font-heading font-bold text-slate-100 text-sm border-b border-brand-border pb-3 flex items-center justify-between">
              <span>AI Diagnostics</span>
              {predictionData && (
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wider uppercase ${predictionData.prediction === 'POSITIVE' ? 'bg-brand-pathology/15 border border-brand-pathology/30 text-brand-pathology' : 'bg-brand-healthy/15 border border-brand-healthy/30 text-brand-healthy'}`}>
                  {predictionData.prediction}
                </span>
              )}
            </h3>

            {/* Viewport states */}
            {predictionLoading ? (
              <div className="text-center py-12 space-y-3">
                <div className="w-8 h-8 rounded-full border-2 border-slate-900 border-t-brand-cyan animate-spin mx-auto"></div>
                <p className="text-xs text-brand-textMuted">Checking prediction records...</p>
              </div>
            ) : isPredicting ? (
              <div className="text-center py-12 space-y-3">
                <div className="w-8 h-8 rounded-full border-2 border-slate-900 border-t-brand-violet animate-spin mx-auto"></div>
                <p className="text-xs text-brand-textMuted font-semibold">Running model inference...</p>
                <p className="text-[10px] text-slate-500 max-w-xs mx-auto leading-relaxed">Computing attention layers and enqueuing active learning weights...</p>
              </div>
            ) : predictionData ? (
              <div className="space-y-5">
                {/* Result Prob */}
                <div className="text-center p-4 bg-slate-950/40 rounded-xl border border-brand-border flex flex-col items-center justify-center">
                  <span className="font-heading font-black text-4xl text-white">
                    {Math.round(predictionData.probability * 100)}%
                  </span>
                  <span className="text-[10px] uppercase font-bold text-brand-textMuted tracking-wider mt-1">
                    AI Pathology Probability
                  </span>
                </div>

                {/* Narrative */}
                <div className="space-y-1">
                  <span className="text-[10px] uppercase font-bold text-brand-textMuted tracking-wider">AI Clinical Explanation</span>
                  <p className="text-xs leading-relaxed text-slate-300 font-medium bg-slate-950/30 border border-brand-border/40 p-3 rounded-lg">
                    {predictionData.text_justification}
                  </p>
                </div>

                {/* Secondary downloads */}
                <div className="space-y-2 pt-3 border-t border-brand-border">
                  {predictionData.sc_url && (
                    <a
                      href={formatUrl(predictionData.sc_url)}
                      download
                      className="flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl border border-brand-border bg-slate-950 hover:bg-slate-905 text-xs font-semibold text-slate-300 transition-colors uppercase tracking-wider text-center"
                    >
                      <FileDown className="w-4 h-4 text-brand-cyan" />
                      <span>Download SC DICOM</span>
                    </a>
                  )}
                  {predictionData.sr_url && (
                    <a
                      href={formatUrl(predictionData.sr_url)}
                      download
                      className="flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl border border-brand-border bg-slate-950 hover:bg-slate-905 text-xs font-semibold text-slate-300 transition-colors uppercase tracking-wider text-center"
                    >
                      <FileText className="w-4 h-4 text-brand-violet" />
                      <span>Download SR DICOM</span>
                    </a>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center py-8 space-y-4">
                <p className="text-xs text-brand-textMuted leading-relaxed">
                  This study has not yet been analyzed by the PneumoDex diagnostic engine.
                </p>
                <button
                  onClick={handleRunPrediction}
                  className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-brand-cyan to-brand-violet hover:brightness-110 text-slate-955 font-extrabold uppercase text-xs tracking-wider flex items-center justify-center gap-2 shadow-lg shadow-brand-cyan/20 transition-all duration-300"
                >
                  <Play className="w-3.5 h-3.5 fill-slate-950 text-slate-950" />
                  <span>Run Diagnostics</span>
                </button>
              </div>
            )}
          </div>
        </div>

      </div>

    </div>
  );
}
