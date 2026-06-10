import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { 
  Activity, 
  BrainCircuit, 
  Database, 
  Scale, 
  TrendingUp, 
  ShieldCheck, 
  Sliders, 
  X,
  AlertCircle,
  CheckCircle,
  Info
} from 'lucide-react';

export default function Layout({ children }) {
  const { notificationQueue, removeNotification } = useApp();
  const location = useLocation();

  const navigation = [
    { name: 'X-Ray Acquisition', path: '/', icon: BrainCircuit },
    { name: 'Studies Registry', path: '/studies', icon: Database },
    { name: 'Fairness Analysis', path: '/fairness', icon: Scale },
    { name: 'Drift Monitoring', path: '/drift', icon: TrendingUp },
    { name: 'Ledger Audit', path: '/audit', icon: ShieldCheck },
    { name: 'Control Panel', path: '/settings', icon: Sliders },
  ];

  return (
    <div className="min-h-screen flex bg-brand-bg font-sans selection:bg-brand-cyan/20 selection:text-white relative">
      
      {/* Sidebar Navigation */}
      <aside className="w-80 border-r border-brand-border bg-slate-950/40 backdrop-blur-md flex flex-col fixed h-screen z-20">
        
        {/* Brand Container */}
        <div className="p-6 border-b border-brand-border flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-cyan to-brand-violet flex items-center justify-center shadow-lg shadow-brand-cyan/25 animate-pulse">
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
        <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto">
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) => `
                  flex items-center gap-3.5 px-4 py-3.5 rounded-xl text-sm font-medium transition-all duration-300
                  ${isActive 
                    ? 'bg-gradient-to-r from-brand-cyan/10 to-brand-violet/5 border border-brand-cyan/30 text-white shadow-inner shadow-brand-cyan/5' 
                    : 'text-slate-400 hover:text-white hover:bg-slate-900/40 border border-transparent'}
                `}
              >
                {({ isActive }) => (
                  <>
                    <Icon className={`w-5 h-5 transition-transform duration-300 group-hover:scale-110 ${isActive ? 'text-brand-cyan filter drop-shadow-[0_0_8px_rgba(14,165,233,0.5)]' : 'text-slate-500'}`} />
                    <span>{item.name}</span>
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
            <span className="font-semibold text-slate-300">ViT-B/16 (LoRA)</span>
          </div>
          <div className="flex items-center justify-between text-xs text-brand-textMuted">
            <span>Core Version</span>
            <span className="font-semibold text-slate-300">v1.2.0-onnx</span>
          </div>
          <div className="flex items-center gap-2 mt-2 text-xs text-brand-healthy">
            <span className="w-2 h-2 rounded-full bg-brand-healthy animate-ping"></span>
            <span className="font-medium tracking-wide">Celery Worker Ready</span>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 pl-80 min-h-screen flex flex-col">
        
        {/* Sticky Header */}
        <header className="h-20 border-b border-brand-border bg-brand-bg/85 backdrop-blur-md px-8 flex items-center justify-between sticky top-0 z-10">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-brand-cyan/85">
              Current Session
            </span>
            <h2 className="font-heading font-bold text-lg text-white">
              {navigation.find(n => n.path === location.pathname)?.name || 'Study Viewer'}
            </h2>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/60 border border-brand-border text-xs text-slate-300">
              <Activity className="w-3.5 h-3.5 text-brand-cyan" />
              <span>Host Node: <span className="font-semibold">Local-GPU-0</span></span>
            </div>
            <div className="w-8 h-8 rounded-full bg-brand-violet/20 border border-brand-violet/40 flex items-center justify-center">
              <span className="text-xs font-semibold text-brand-violet">MD</span>
            </div>
          </div>
        </header>

        {/* Page Container */}
        <main className="flex-1 p-8 overflow-y-auto max-w-[1600px] w-full mx-auto">
          {children}
        </main>
      </div>

      {/* Notification Toast Stack Overlay */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-3 max-w-md w-full">
        {notificationQueue.map((n) => (
          <div 
            key={n.id} 
            className="glass-panel p-4 rounded-xl border-slate-700 flex items-start gap-3 shadow-2xl relative animate-slide-up"
          >
            {n.type === 'error' && <AlertCircle className="w-5 h-5 text-brand-pathology shrink-0" />}
            {n.type === 'success' && <CheckCircle className="w-5 h-5 text-brand-healthy shrink-0" />}
            {n.type === 'info' && <Info className="w-5 h-5 text-brand-cyan shrink-0" />}
            
            <div className="flex-1 text-sm leading-relaxed pr-6 text-slate-200">
              {n.message}
            </div>

            <button 
              onClick={() => removeNotification(n.id)}
              className="absolute top-3 right-3 text-slate-500 hover:text-slate-200 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        ))}
      </div>
      
    </div>
  );
}
