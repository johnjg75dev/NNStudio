import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';
import Icon from '../components/Icon';

const ToastContext = createContext(null);

let nextId = 1;

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const [dialog, setDialog] = useState(null);
  const timers = useRef(new Map());

  const dismiss = useCallback((id) => {
    setToasts((list) => list.map((t) => (t.id === id ? { ...t, leaving: true } : t)));
    const timer = setTimeout(() => {
      setToasts((list) => list.filter((t) => t.id !== id));
      timers.current.delete(id);
    }, 220);
    timers.current.set(id, timer);
  }, []);

  const push = useCallback(
    (message, { tone = 'info', title = '', ttl = 4200 } = {}) => {
      const id = nextId++;
      setToasts((list) => [...list.slice(-3), { id, message, tone, title, leaving: false }]);
      if (ttl) {
        const timer = setTimeout(() => dismiss(id), ttl);
        timers.current.set(id, timer);
      }
      return id;
    },
    [dismiss],
  );

  const toast = useMemo(
    () => ({
      push,
      dismiss,
      info: (message, opts) => push(message, { ...opts, tone: 'info' }),
      success: (message, opts) => push(message, { ...opts, tone: 'pos' }),
      warn: (message, opts) => push(message, { ...opts, tone: 'warn' }),
      error: (message, opts) => push(message, { ...opts, tone: 'neg', ttl: 6500 }),
    }),
    [push, dismiss],
  );

  /** Promise-based replacement for window.confirm. */
  const confirm = useCallback(
    ({ title = 'Are you sure?', message = '', confirmLabel = 'Confirm', cancelLabel = 'Cancel', tone = 'danger' } = {}) =>
      new Promise((resolve) => {
        setDialog({ title, message, confirmLabel, cancelLabel, tone, resolve });
      }),
    [],
  );

  const closeDialog = useCallback((result) => {
    setDialog((d) => {
      d?.resolve(result);
      return null;
    });
  }, []);

  const value = useMemo(() => ({ toast, confirm }), [toast, confirm]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((t) => (
          <Toast key={t.id} toast={t} onDismiss={() => dismiss(t.id)} />
        ))}
      </div>
      {dialog && <ConfirmDialog dialog={dialog} onClose={closeDialog} />}
    </ToastContext.Provider>
  );
}

const ICONS = {
  pos: 'check',
  neg: 'error',
  warn: 'alert',
  info: 'info',
};

function Toast({ toast: t, onDismiss }) {
  return (
    <div className={`toast toast--${t.tone}`} data-leaving={t.leaving ? 'true' : 'false'}>
      <Icon name={ICONS[t.tone] || 'info'} className="toast__icon" size={17} />
      <div className="toast__body">
        {t.title && <div className="toast__title">{t.title}</div>}
        <div className="toast__msg">{t.message}</div>
      </div>
      <button className="tool-btn" onClick={onDismiss} aria-label="Dismiss notification">
        <Icon name="close" size={13} />
      </button>
    </div>
  );
}

function ConfirmDialog({ dialog, onClose }) {
  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose(false)}>
      <div className="modal modal--narrow" role="alertdialog" aria-modal="true">
        <div className="modal__head">
          <h3>{dialog.title}</h3>
        </div>
        {dialog.message && (
          <div className="modal__body">
            <p className="dim" style={{ fontSize: 'var(--fs-md)' }}>
              {dialog.message}
            </p>
          </div>
        )}
        <div className="modal__foot">
          <button className="btn btn--ghost" onClick={() => onClose(false)}>
            {dialog.cancelLabel}
          </button>
          <button
            className={`btn ${dialog.tone === 'danger' ? 'btn--danger' : 'btn--primary'}`}
            onClick={() => onClose(true)}
            autoFocus
          >
            {dialog.confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used inside <ToastProvider>');
  return ctx.toast;
}

export function useConfirm() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useConfirm must be used inside <ToastProvider>');
  return ctx.confirm;
}
