import React, { useState, useEffect, useCallback, useRef } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { 
  Activity, 
  BrainCircuit, 
  Database, 
  Scale, 
  TrendingUp, 
  ShieldCheck, 
  Syringe,
  Sliders, 
  X,
  AlertCircle,
  CheckCircle,
  Info,
  Menu,
  ChevronRight,
} from 'lucide-react';

const APP_VERSION = import.meta.env.VITE_APP_VERSION || 'v1.3.0';

export default function Layout({ children }) {
  const { settings, serverStatus, notificationQueue, removeNotification } = useApp();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Close sidebar on route change (mobile)
  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname]);

  const navigation = [
    { name: 'X-Ray Acquisition', path: '/', icon: BrainCircuit },
    { name: 'Studies Registry', path: '/studies', icon: Database },
    { name: 'Active Learning', path: '/active-learning', icon: Syringe },
    { name: 'Fairness Analysis', path: '/fairness', icon: Scale },
    { name: 'Drift Monitoring', path: '/drift', icon: TrendingUp },
    { name: 'Ledger Audit', path: '/audit', icon: ShieldCheck },
    { name: 'Control Panel', path: '/settings', icon: Sliders },
  ];

  const getStatusColorClass = () => {
    if (serverStatus === 'Celery Worker Ready') return 'text-brand-healthy';
    if (serverStatus === 'Celery Offline') return 'text-amber-500';
    if (serverStatus === 'Checking...') return 'text-brand-textMuted';
    return 'text-brand-pathology';
  };

  const getStatusDotColorClass = () => {
    if (serverStatus === 'Celery Worker Ready') return 'bg-brand-healthy';
    if (serverStatus === 'Celery Offline') return 'bg-amber-500';
    if (serverStatus === 'Checking...') return 'bg-brand-textMuted';
    return 'bg-brand-pathology';
  };

  // Find current page name — handle dynamic routes
  const getPageName = () => {
    if (location.pathname.startsWith('/reports/')) return 'Clinical Report';
    if (location.pathname.startsWith('/studies/') && location.pathname.length > '/studies/'.length) return 'Study Viewer';
    return navigation.find(n => n.path === location.pathname)?.name || 'PneumoDex';
  };

  const SidebarContent = () => (
    <>
      {/* Brand Container */}
      <div className="p-6 border-b border-brand-border flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-cyan to-brand-violet flex items-center justify-center shadow-lg shadow-brand-cyan/25">
          <span className="text-xl font-bold text-slate-950">🫁</span>
        </div>
        <div>
          <h1 className="font-heading font-extrabold text-xl bg-gradient-to-r from-white via-slate-200 to-brand-cyan bg-clip-text text-transparent leading-tight tracking-tight">
            PneumoDex
          </h1>
          <span className="text-[10px] uppercase font-bold tracking-widest text-brand-cyan/85">
            AI Diagnostic Suite
          </span>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 px-4 py-6 space-y-1 overflow-y-auto">
        {navigation.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) => `
                flex items-center gap-3.5 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 group
                ${isActive 
                  ? 'bg-gradient-to-r from-brand-cyan/10 to-brand-violet/5 border border-brand-cyan/30 text-white shadow-inner shadow-brand-cyan/5' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-900/40 border border-transparent'}
              `}
            >
              {({ isActive }) => (
                <>
                  <Icon className={`w-5 h-5 transition-all duration-200 ${isActive ? 'text-brand-cyan drop-shadow-[0_0_8px_rgba(14,165,233,0.5)]' : 'text-slate-500 group-hover:text-slate-300'}`} />
                  <span className="flex-1">{item.name}</span>
                  {isActive && <ChevronRight className="w-3.5 h-3.5 text-brand-cyan/50" />}
                </>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Sidebar Footer System Info */}
      <div className="p-5 border-t border-brand-border bg-slate-950/20 flex flex-col gap-2">
        <div className="flex items-center justify-between text-xs text-brand-textMuted">
          <span>Model Engine</span>
          <span className="font-semibold text-slate-300">
            {settings.modelType === 'resnet' ? 'ResNet-50 (Baseline)' 
              : settings.modelType === 'medfound' ? 'BioViL-T (Med)' 
              : 'ViT-B/16 (LoRA)'}
          </span>
        </div>
        <div className="flex items-center justify-between text-xs text-brand-textMuted">
          <span>Core Version</span>
          <span className="font-semibold text-slate-300">{APP_VERSION}</span>
        </div>
        <div className={`flex items-center gap-2 mt-1 text-xs ${getStatusColorClass()}`}>
          <span className={`w-2 h-2 rounded-full ${serverStatus === 'Celery Worker Ready' ? 'animate-ping' : ''} ${getStatusDotColorClass()}`}></span>
          <span className="font-medium tracking-wide">{serverStatus}</span>
        </div>
      </div>
    </>
  );

  return (
    <div className="min-h-screen flex bg-brand-bg font-sans selection:bg-brand-cyan/20 selection:text-white relative">
      
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/60 z-30 lg:hidden" 
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar — fixed on desktop, slide-in on mobile */}
      <aside className={`
        w-72 border-r border-brand-border bg-slate-950/95 backdrop-blur-md flex flex-col fixed h-screen z-40 transition-transform duration-300
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        lg:translate-x-0 lg:z-20
      `}>
        <SidebarContent />
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 lg:pl-72 min-h-screen flex flex-col">
        
        {/* Sticky Header */}
        <header className="h-16 border-b border-brand-border bg-brand-bg/85 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-10">
          <div className="flex items-center gap-4">
            {/* Hamburger — mobile only */}
            <button
              className="lg:hidden p-2 rounded-lg border border-brand-border hover:bg-slate-900 text-slate-300 transition-colors"
              onClick={() => setSidebarOpen(prev => !prev)}
              aria-label="Toggle navigation"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-brand-cyan/85">
                Current Session
              </span>
              <h2 className="font-heading font-bold text-base lg:text-lg text-white leading-tight">
                {getPageName()}
              </h2>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/60 border border-brand-border text-xs text-slate-300">
              <Activity className="w-3.5 h-3.5 text-brand-cyan" />
              <span>ONNX CPU Inference</span>
            </div>
            <div className="w-8 h-8 rounded-full bg-brand-violet/20 border border-brand-violet/40 flex items-center justify-center" title="Radiologist Session">
              <span className="text-xs font-semibold text-brand-violet">MD</span>
            </div>
          </div>
        </header>

        {/* Page Container */}
        <main className="flex-1 p-6 lg:p-8 overflow-y-auto max-w-[1600px] w-full mx-auto">
          {children}
        </main>
      </div>

      {/* Notification Toast Stack Overlay */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-3 max-w-sm w-full pointer-events-none">
        {notificationQueue.map((n) => (
          <div 
            key={n.id} 
            className="glass-panel p-4 rounded-xl flex items-start gap-3 shadow-2xl relative animate-slide-up pointer-events-auto"
          >
            {n.type === 'error' && <AlertCircle className="w-5 h-5 text-brand-pathology shrink-0 mt-0.5" />}
            {n.type === 'success' && <CheckCircle className="w-5 h-5 text-brand-healthy shrink-0 mt-0.5" />}
            {n.type === 'info' && <Info className="w-5 h-5 text-brand-cyan shrink-0 mt-0.5" />}
            
            <div className="flex-1 text-xs leading-relaxed pr-6 text-slate-200">
              {n.message}
            </div>

            <button 
              onClick={() => removeNotification(n.id)}
              className="absolute top-3 right-3 text-slate-500 hover:text-slate-200 transition-colors"
              aria-label="Dismiss notification"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        ))}
      </div>
      
    </div>
  );
}
