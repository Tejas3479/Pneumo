import React, { createContext, useState, useContext, useEffect } from 'react';
import { updateBaseURL } from '../api/client';

const AppContext = createContext(null);

export const AppContextProvider = ({ children }) => {
  const [activeStudy, setActiveStudy] = useState(null);
  const [settings, setSettings] = useState(() => {
    const saved = localStorage.getItem('pneumodex_settings');
    return saved ? JSON.parse(saved) : { modelType: 'vit', serverUrl: '' };
  });
  const [notificationQueue, setNotificationQueue] = useState([]);

  useEffect(() => {
    localStorage.setItem('pneumodex_settings', JSON.stringify(settings));
    updateBaseURL(settings.serverUrl);
  }, [settings]);

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
