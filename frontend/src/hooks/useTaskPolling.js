import { useState, useEffect, useRef } from 'react';
import { api } from '../api/client';

export const useTaskPolling = (taskId) => {
  const [status, setStatus] = useState('IDLE'); // IDLE, PENDING, SUCCESS, FAILED
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const timeoutRef = useRef(null);
  const pollCountRef = useRef(0);

  useEffect(() => {
    if (!taskId) {
      setStatus('IDLE');
      setResult(null);
      setError(null);
      return;
    }

    setStatus('PENDING');
    setResult(null);
    setError(null);
    pollCountRef.current = 0;

    let delay = 1000; // initial delay: 1s

    const poll = async () => {
      // Max 40 attempts with exponential backoff capped at 8s ≈ ~5 min max wait
      if (pollCountRef.current >= 40) {
        setStatus('FAILED');
        setError('Task polling timed out. The worker may still be processing — check Celery logs.');
        return;
      }

      try {
        const data = await api.pollTask(taskId);

        if (data.status === 'PENDING' || data.status === 'STARTED') {
          pollCountRef.current += 1;
          delay = Math.min(delay * 1.5, 8000); // Exponential backoff up to 8s
          timeoutRef.current = setTimeout(poll, delay);
        } else if (data.status === 'SUCCESS') {
          setResult(data.result);
          setStatus('SUCCESS');
        } else if (data.status === 'FAILED') {
          setError(data.error || 'Celery task execution failed on worker.');
          setStatus('FAILED');
        } else {
          pollCountRef.current += 1;
          timeoutRef.current = setTimeout(poll, delay);
        }
      } catch (err) {
        pollCountRef.current += 1;
        delay = Math.min(delay * 1.5, 8000);
        timeoutRef.current = setTimeout(poll, delay);
      }
    };

    timeoutRef.current = setTimeout(poll, delay);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [taskId]);

  return { status, result, error };
};
