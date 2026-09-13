import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import Icon from './Icon';
import { useClickOutside } from '../lib/hooks';

/* ═══════════════════════════ Buttons ═══════════════════════════ */

export function Button({
  as: Tag = 'button',
  variant = 'default',
  size = 'md',
  icon,
  iconRight,
  block,
  iconOnly,
  loading,
  className = '',
  children,
  ...rest
}) {
  const cls = [
    'btn',
    variant !== 'default' ? `btn--${variant}` : '',
    size !== 'md' ? `btn--${size}` : '',
    block ? 'btn--block' : '',
    iconOnly ? 'btn--icon' : '',
    className,
  ]
    .filter(Boolean)
    .join(' ');
  return (
    <Tag className={cls} disabled={rest.disabled || loading} {...rest}>
      {loading ? <span className="spinner" /> : icon ? <Icon name={icon} /> : null}
      {children}
      {iconRight ? <Icon name={iconRight} /> : null}
    </Tag>
  );
}

/* ═══════════════════════════ Surfaces ═══════════════════════════ */

export function Card({ title, subtitle, actions, children, className = '', padded = true, headless = false }) {
  return (
    <section className={`card ${className}`}>
      {!headless && (title || actions) && (
        <header className="card__head">
          <div className="grow">
            {title && <h3>{title}</h3>}
            {subtitle && (
              <p className="tiny muted" style={{ marginTop: 2 }}>
                {subtitle}
              </p>
            )}
          </div>
          {actions}
        </header>
      )}
      <div className={padded ? 'card__body' : undefined} style={padded ? undefined : { padding: 0 }}>
        {children}
      </div>
    </section>
  );
}

export function SectionTitle({ children, right }) {
  return (
    <div className="section-title">
      <span>{children}</span>
      {right}
    </div>
  );
}

export function Info({ tone = '', children, className = '' }) {
  return <div className={`info ${tone ? `info--${tone}` : ''} ${className}`}>{children}</div>;
}

export function EmptyState({ icon = 'info', title, message, children, action }) {
  return (
    <div className="empty">
      <div className="empty__icon">
        <Icon name={icon} size={22} />
      </div>
      {title && <div className="empty__title">{title}</div>}
      {(children || message) && <div className="empty__text">{children || message}</div>}
      {action && <div style={{ marginTop: 'var(--sp-2)' }}>{action}</div>}
    </div>
  );
}

/* ═══════════════════════════ Form fields ═══════════════════════════ */

export function Field({ label, hint, tip, htmlFor, children, className = '', inline = false, style, ...rest }) {
  return (
    <div
      className={`field ${className}`}
      style={{ ...(inline ? { flexDirection: 'row', alignItems: 'center', gap: 8 } : null), ...style }}
      {...rest}
    >
      {label && (
        <label className="field__label" htmlFor={htmlFor}>
          <span>{label}</span>
          {tip && <Tooltip content={tip} />}
        </label>
      )}
      {children}
      {hint && <span className="field__hint">{hint}</span>}
    </div>
  );
}

export function Select({ value, onChange, options, children, className = '', ...rest }) {
  return (
    <select
      className={`select ${className}`}
      value={value ?? ''}
      onChange={(e) => onChange?.(e.target.value, e)}
      {...rest}
    >
      {children ||
        (options || []).map((o) => (
          <option key={o.value ?? o.key} value={o.value ?? o.key}>
            {o.label}
          </option>
        ))}
    </select>
  );
}

export function TextInput({ value, onChange, className = '', ...rest }) {
  return (
    <input
      className={`input ${className}`}
      value={value ?? ''}
      onChange={(e) => onChange?.(e.target.value, e)}
      {...rest}
    />
  );
}

export function NumberInput({ value, onChange, className = '', ...rest }) {
  return (
    <input
      type="number"
      className={`input ${className}`}
      value={value ?? 0}
      onChange={(e) => onChange?.(e.target.value === '' ? '' : Number(e.target.value), e)}
      {...rest}
    />
  );
}

export function Slider({ value, min, max, step = 1, onChange, onCommit, format, label, tip, id, className = '' }) {
  const pct = ((Number(value) - min) / (max - min || 1)) * 100;
  const autoId = useId();
  const inputId = id || autoId;
  return (
    <div className={`field ${className}`.trim()}>
      {label && (
        <div className="row row--between">
          <label className="field__label" htmlFor={inputId}>
            <span>{label}</span>
            {tip && <Tooltip content={tip} />}
          </label>
          <span className="badge badge--mono badge--accent">{format ? format(value) : value}</span>
        </div>
      )}
      <input
        id={inputId}
        className="range"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ '--fill': `${Math.max(0, Math.min(100, pct))}%` }}
        onChange={(e) => onChange?.(Number(e.target.value), e)}
        onPointerUp={() => onCommit?.(Number(value))}
        onKeyUp={() => onCommit?.(Number(value))}
      />
    </div>
  );
}

export function Switch({ checked, onChange, label, tip, disabled }) {
  return (
    <label className="switch" style={disabled ? { opacity: 0.5, cursor: 'not-allowed' } : undefined}>
      <input
        type="checkbox"
        checked={!!checked}
        disabled={disabled}
        onChange={(e) => onChange?.(e.target.checked, e)}
      />
      <span className="switch__track" />
      <span>{label}</span>
      {tip && <Tooltip content={tip} />}
    </label>
  );
}

export function Checkbox({ checked, onChange, label, tip, disabled }) {
  return (
    <label className="check" style={disabled ? { opacity: 0.5 } : undefined}>
      <input
        type="checkbox"
        checked={!!checked}
        disabled={disabled}
        onChange={(e) => onChange?.(e.target.checked, e)}
      />
      <span>{label}</span>
      {tip && <Tooltip content={tip} />}
    </label>
  );
}

export function Segmented({ value, onChange, options, fill = false, size = 'md' }) {
  return (
    <div className={`seg ${fill ? 'seg--fill' : ''}`} role="tablist">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="tab"
          aria-selected={value === o.value}
          className="seg__btn"
          data-active={value === o.value ? 'true' : 'false'}
          onClick={() => onChange?.(o.value)}
          title={o.title}
          style={fill ? { flex: 1 } : undefined}
        >
          {o.icon && <Icon name={o.icon} size={12} style={{ marginRight: 4 }} />}
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Tabs({ value, onChange, tabs, fill = false, className = '' }) {
  return (
    <div className={`tabs ${fill ? 'tabs--fill' : ''} ${className}`} role="tablist">
      {tabs.map((t) => (
        <button
          key={t.value}
          type="button"
          role="tab"
          aria-selected={value === t.value}
          className="tab"
          data-active={value === t.value ? 'true' : 'false'}
          onClick={() => onChange?.(t.value)}
          title={t.title}
        >
          {t.icon && <Icon name={t.icon} size={13} />}
          {t.label}
          {t.count !== undefined && t.count !== null && (
            <span className="badge badge--mono" style={{ height: 16, padding: '0 5px' }}>
              {t.count}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}

export function SearchInput({ value, onChange, placeholder = 'Search…', className = '' }) {
  return (
    <div className={`search ${className}`}>
      <Icon name="search" />
      <input
        className="input"
        type="search"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange?.(e.target.value)}
      />
    </div>
  );
}

/* ═══════════════════════════ Display ═══════════════════════════ */

export function Badge({ tone = '', mono, children, className = '', title, ...rest }) {
  return (
    <span
      className={`badge ${tone ? `badge--${tone}` : ''} ${mono ? 'badge--mono' : ''} ${className}`}
      title={title}
      {...rest}
    >
      {children}
    </span>
  );
}

export function StatusPill({ tone = '', children, dot = true }) {
  return (
    <span className="pill-status" data-tone={tone}>
      {dot && <span className="pill-status__dot" />}
      {children}
    </span>
  );
}

export function Metric({ label, value, tone = '', title, spark }) {
  return (
    <div className={`metric ${tone ? `metric--${tone}` : ''}`} title={title}>
      <div className="metric__k">{label}</div>
      <div className="metric__v">{value}</div>
      {spark}
    </div>
  );
}

export function ProgressBar({ value, max = 100 }) {
  const pct = Math.max(0, Math.min(100, (value / (max || 1)) * 100));
  return (
    <div className="progress">
      <div className="progress__bar" style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Spinner({ size = 'md', label }) {
  return (
    <div className="row" style={{ gap: 10, color: 'var(--text-3)' }}>
      <span className={`spinner ${size === 'lg' ? 'spinner--lg' : ''}`} />
      {label && <span className="tiny">{label}</span>}
    </div>
  );
}

/* ═══════════════════════════ Tooltip ═══════════════════════════ */

export function Tooltip({ content, html, children, side = 'right' }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const triggerRef = useRef(null);

  const show = useCallback(() => {
    const el = triggerRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setPos({ x: r.left + r.width / 2, y: r.top });
    setOpen(true);
  }, []);

  return (
    <>
      <span
        ref={triggerRef}
        className="tip-trigger"
        tabIndex={0}
        onMouseEnter={show}
        onMouseLeave={() => setOpen(false)}
        onFocus={show}
        onBlur={() => setOpen(false)}
        aria-label="More information"
      >
        ?
      </span>
      {open &&
        createPortal(
          <TooltipBubble x={pos.x} y={pos.y} content={content} html={html} side={side} />,
          document.body,
        )}
      {children}
    </>
  );
}

function TooltipBubble({ x, y, content, html, side }) {
  const ref = useRef(null);
  const [style, setStyle] = useState({ left: x, top: y, visibility: 'hidden' });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    let left = x - r.width / 2;
    let top = y - r.height - 8;
    if (side === 'right') {
      left = x + 14;
      top = y - 6;
    }
    left = Math.max(8, Math.min(left, window.innerWidth - r.width - 8));
    if (top < 8) top = y + 18;
    top = Math.min(top, window.innerHeight - r.height - 8);
    setStyle({ left, top, visibility: 'visible' });
  }, [x, y, side]);

  return (
    <div className="tooltip" ref={ref} style={style} role="tooltip">
      {html ? <span dangerouslySetInnerHTML={{ __html: html }} /> : content}
    </div>
  );
}

/** Rich description block used for registry entries (HTML from the backend). */
export function HtmlText({ html, className = '' }) {
  if (!html) return null;
  return <div className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

/* ═══════════════════════════ Modal ═══════════════════════════ */

export function Modal({ open, onClose, title, subtitle, icon, children, footer, size = 'md' }) {
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') onClose?.();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;
  const widthClass = size === 'wide' ? 'modal--wide' : size === 'narrow' ? 'modal--narrow' : '';

  return createPortal(
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose?.()}>
      <div className={`modal ${widthClass}`} role="dialog" aria-modal="true" aria-label={title}>
        <div className="modal__head">
          {icon && (
            <span
              style={{
                display: 'grid',
                placeItems: 'center',
                width: 32,
                height: 32,
                borderRadius: 'var(--r-md)',
                background: 'var(--accent-soft)',
                color: 'var(--accent)',
              }}
            >
              <Icon name={icon} size={17} />
            </span>
          )}
          <div className="grow">
            <h3>{title}</h3>
            {subtitle && <p>{subtitle}</p>}
          </div>
          <button className="modal__close" onClick={onClose} aria-label="Close dialog">
            <Icon name="close" size={15} />
          </button>
        </div>
        <div className="modal__body">{children}</div>
        {footer && <div className="modal__foot">{footer}</div>}
      </div>
    </div>,
    document.body,
  );
}

/* ═══════════════════════════ Popover ═══════════════════════════ */

export function Popover({ trigger, children, align = 'right', width = 240 }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useClickOutside(ref, () => setOpen(false), open);
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <div onClick={() => setOpen((o) => !o)}>{trigger(open)}</div>
      {open && (
        <div
          className="card"
          style={{
            position: 'absolute',
            top: 'calc(100% + 6px)',
            [align === 'right' ? 'right' : 'left']: 0,
            width,
            zIndex: 80,
            padding: 'var(--sp-2)',
            boxShadow: 'var(--shadow-3)',
          }}
        >
          {typeof children === 'function' ? children(() => setOpen(false)) : children}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════ Accordion ═══════════════════════════ */

export function Accordion({ step, label, meta, open, onToggle, done, children, defaultOpen = false }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const controlled = open !== undefined;
  const expanded = controlled ? open : isOpen;
  const toggle = () => (controlled ? onToggle?.(!expanded) : setIsOpen((v) => !v));

  return (
    <section className="acc" data-open={expanded ? 'true' : 'false'} data-done={done ? 'true' : 'false'}>
      <button type="button" className="acc__head" onClick={toggle} aria-expanded={expanded}>
        {step !== undefined && (
          <span className="acc__step">{done ? <Icon name="check" size={11} strokeWidth={3} /> : step}</span>
        )}
        <span className="acc__label">{label}</span>
        {meta && <span className="acc__meta">{meta}</span>}
        <Icon name="chevronRight" className="acc__chev" />
      </button>
      {expanded && <div className="acc__body">{children}</div>}
    </section>
  );
}
