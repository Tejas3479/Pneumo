import React, { createContext, useState, useContext, useEffect } from 'react';
import { updateBaseURL } from '../api/client';

const AppContext = createContext(null);

export const AppContextProvider = ({ children }) => {
  const [activeStudy, setActiveStudy] = useState(null);
  const [settings, setSettings] = useState(() => {
    const saved = localStorage.getItem('pneumodex_settings');
    if (saved) {
      const parsed = JSON.parse(saved);
      return { modelType: 'vit', serverUrl: '', apiKey: '', ...parsed };
    }
    return { modelType: 'vit', serverUrl: '', apiKey: '' };
  });
  const [serverStatus, setServerStatus] = useState('Checking...');
  const [notificationQueue, setNotificationQueue] = useState([]);

  useEffect(() => {
    localStorage.setItem('pneumodex_settings', JSON.stringify(settings));
    updateBaseURL(settings.serverUrl);
  }, [settings]);

  useEffect(() => {
    const checkStatus = async () => {
      const baseUrl = settings.serverUrl || window.location.origin;
      const url = `${baseUrl.replace(/\/$/, '')}/ready`;
      try {
        const res = await fetch(url);
        if (res.ok) {
          setServerStatus('Celery Worker Ready');
        } else {
          setServerStatus('Celery Offline');
        }
      } catch (err) {
        setServerStatus('Server Offline');
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 10000);
    return () => clearInterval(interval);
  }, [settings.serverUrl]);

  const addNotification = (message, type = 'info') => {
    const id = Date.now();
    setNotificationQueue((prev) => [...prev, { id, message, type }]);
    
    // Auto-remove notification after 5 seconds
    setTimeout(() => {
      setNotificationQueue((prev) => prev.filter((n) => n.id !== id));
    }, 5000);
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
