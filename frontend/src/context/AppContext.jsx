import React, { createContext, useState, useContext, useEffect, useRef } from 'react';
import { updateBaseURL } from '../api/client';

const AppContext = createContext(null);

export const AppContextProvider = ({ children }) => {
  const [activeStudy, setActiveStudy] = useState(null);
  const [settings, setSettings] = useState(() => {
    try {
      const saved = localStorage.getItem('pneumodex_settings');
      if (saved) {
        const parsed = JSON.parse(saved);
        return { modelType: 'vit', serverUrl: '', apiKey: '', ...parsed };
      }
    } catch (e) {
      console.warn('[AppContext] Failed to parse saved settings from localStorage:', e);
    }
    return { modelType: 'vit', serverUrl: '', apiKey: '' };
  });
  const [serverStatus, setServerStatus] = useState('Checking...');
  const [notificationQueue, setNotificationQueue] = useState([]);
  const abortControllerRef = useRef(null);

  useEffect(() => {
    try {
      localStorage.setItem('pneumodex_settings', JSON.stringify(settings));
    } catch (e) {
      console.warn('[AppContext] Failed to save settings to localStorage:', e);
    }
    updateBaseURL(settings.serverUrl);
  }, [settings]);

  useEffect(() => {
    const checkStatus = async () => {
      // Cancel previous in-flight check
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();

      const baseUrl = settings.serverUrl || window.location.origin;
      const url = `${baseUrl.replace(/\/$/, '')}/ready`;
      try {
        const res = await fetch(url, { signal: abortControllerRef.current.signal });
        if (res.ok) {
          setServerStatus('Celery Worker Ready');
        } else {
          setServerStatus('Celery Offline');
        }
      } catch (err) {
        if (err.name === 'AbortError') return; // Ignore aborted requests
        setServerStatus('Server Offline');
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => {
      clearInterval(interval);
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [settings.serverUrl]);

  const addNotification = (message, type = 'info') => {
    const id = Date.now() + Math.random();
    setNotificationQueue((prev) => [...prev, { id, message, type }]);

    // Auto-remove notification after 5 seconds
    const timer = setTimeout(() => {
      setNotificationQueue((prev) => prev.filter((n) => n.id !== id));
    }, 5000);

    // Return cleanup function
    return () => clearTimeout(timer);
  };

  const removeNotification = (id) => {
    setNotificationQueue((prev) => prev.filter((n) => n.id !== id));
  };

  return (
    <AppContext.Provider
      value={{
        activeStudy,
        setActiveStudy,
        settings,
        setSettings,
        serverStatus,
        notificationQueue,
        addNotification,
        removeNotification,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within AppContextProvider');
  }
  return context;
};
