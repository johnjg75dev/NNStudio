/* SVG icon library — one place for every icon used by the redesigned UI.
 * Usage:  el.innerHTML = ICONS.cube({ size: 16 })
 * All icons share a 24-unit viewBox and use currentColor.
 */
(function () {
  const wrap = (paths, opts = {}) => {
    const size = opts.size ?? 18;
    const sw = opts.stroke ?? 1.75;
    return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${sw}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`;
  };
  const I = {
    home:        o => wrap('<path d="M3 11.5 12 4l9 7.5"/><path d="M5 10v10h14V10"/>', o),
    folder:      o => wrap('<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/>', o),
    plus:        o => wrap('<path d="M12 5v14M5 12h14"/>', o),
    spark:       o => wrap('<path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8"/>', o),
    cube:        o => wrap('<path d="M12 3 3 7.5 12 12l9-4.5L12 3Z"/><path d="M3 7.5V16l9 5 9-5V7.5"/><path d="M12 12v9"/>', o),
    layers:      o => wrap('<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 13 9 5 9-5"/><path d="m3 18 9 5 9-5"/>', o),
    zap:         o => wrap('<path d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z"/>', o),
    flask:       o => wrap('<path d="M9 3h6"/><path d="M10 3v6L4 19a2 2 0 0 0 1.7 3h12.6A2 2 0 0 0 20 19l-6-10V3"/>', o),
    chart:       o => wrap('<path d="M4 19V5"/><path d="M4 19h16"/><path d="m8 16 3-4 3 3 4-7"/>', o),
    package:     o => wrap('<path d="m3 7 9-4 9 4-9 4-9-4Z"/><path d="M3 7v10l9 4 9-4V7"/><path d="M12 11v10"/>', o),
    sliders:     o => wrap('<path d="M4 6h12"/><circle cx="18" cy="6" r="2"/><path d="M4 12h6"/><circle cx="12" cy="12" r="2"/><path d="M4 18h10"/><circle cx="16" cy="18" r="2"/>', o),
    book:        o => wrap('<path d="M4 4h10a4 4 0 0 1 4 4v12H8a4 4 0 0 1-4-4Z"/><path d="M4 4v16"/>', o),
    bug:         o => wrap('<rect x="6" y="9" width="12" height="11" rx="6"/><path d="M9 4 7 6"/><path d="m15 4 2 2"/><path d="M3 15h3"/><path d="M18 15h3"/><path d="M3 11h3"/><path d="M18 11h3"/>', o),
    play:        o => wrap('<path d="M6 4v16l14-8L6 4Z"/>', o),
    pause:       o => wrap('<path d="M7 5v14M17 5v14"/>', o),
    refresh:     o => wrap('<path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 4v5h-5"/>', o),
    save:        o => wrap('<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2Z"/><path d="M7 3v6h10V3"/><path d="M7 13h10v8H7Z"/>', o),
    upload:      o => wrap('<path d="M12 3v12"/><path d="m7 8 5-5 5 5"/><path d="M5 15v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4"/>', o),
    download:    o => wrap('<path d="M12 3v12"/><path d="m7 11 5 5 5-5"/><path d="M5 19h14"/>', o),
    search:      o => wrap('<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>', o),
    settings:    o => wrap('<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3 1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8 1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z"/>', o),
    user:        o => wrap('<path d="M5 21a7 7 0 0 1 14 0"/><circle cx="12" cy="8" r="4"/>', o),
    logout:      o => wrap('<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="m16 17 5-5-5-5"/><path d="M21 12H9"/>', o),
    chevron:     o => wrap('<path d="m9 6 6 6-6 6"/>', o),
    grid:        o => wrap('<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>', o),
    image:       o => wrap('<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/>', o),
    text:        o => wrap('<path d="M4 7h16"/><path d="M4 12h10"/><path d="M4 17h16"/>', o),
    music:       o => wrap('<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>', o),
    waves:       o => wrap('<path d="M2 12c2-3 4-3 6 0s4 3 6 0 4-3 6 0 4 3 6 0"/>', o),
    chat:        o => wrap('<path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4Z"/>', o),
    table:       o => wrap('<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>', o),
    tree:        o => wrap('<circle cx="12" cy="5" r="2"/><circle cx="6" cy="19" r="2"/><circle cx="18" cy="19" r="2"/><path d="M12 7v6M12 13l-6 6M12 13l6 6"/>', o),
    network:     o => wrap('<circle cx="5" cy="12" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="19" cy="18" r="2"/><path d="m7 11 10-4M7 13l10 4"/>', o),
    eye:         o => wrap('<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>', o),
    flame:       o => wrap('<path d="M12 22c4 0 7-3 7-7 0-4-4-6-4-10 0 0-3 1-5 5-1-1-2-2-2-3 0 0-3 3-3 8 0 4 3 7 7 7Z"/>', o),
    target:      o => wrap('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>', o),
    activity:    o => wrap('<path d="M3 12h4l3-9 4 18 3-9h4"/>', o),
    stop:        o => wrap('<rect x="5" y="5" width="14" height="14" rx="2"/>', o),
    rewind:      o => wrap('<path d="m11 19-9-7 9-7v14Z"/><path d="m22 19-9-7 9-7v14Z"/>', o),
    forward:     o => wrap('<path d="m13 5 9 7-9 7V5Z"/><path d="m2 5 9 7-9 7V5Z"/>', o),
    trash:       o => wrap('<path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/>', o),
    copy:        o => wrap('<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v0"/>', o),
    star:        o => wrap('<path d="m12 2 3 7h7l-5.5 4.5L18 22l-6-4-6 4 1.5-8.5L2 9h7l3-7Z"/>', o),
    archive:     o => wrap('<rect x="3" y="3" width="18" height="5" rx="1"/><path d="M5 8v11a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8"/><path d="M10 12h4"/>', o),
    code:        o => wrap('<path d="m9 18-6-6 6-6"/><path d="m15 6 6 6-6 6"/>', o),
    rocket:      o => wrap('<path d="M14 4s5 1 5 7c0 0-3 7-7 7-3 0-5-3-5-3l-4 1 1-4s-3-2-3-5c0-4 7-7 7-7Z"/><circle cx="14" cy="9" r="1.5"/>', o),
    binary:      o => wrap('<rect x="4" y="3" width="6" height="8" rx="1"/><rect x="14" y="13" width="6" height="8" rx="1"/><path d="M14 7h2v2"/><path d="M4 17h2v2"/>', o),
    server:      o => wrap('<rect x="3" y="3" width="18" height="7" rx="2"/><rect x="3" y="14" width="18" height="7" rx="2"/><path d="M7 6h.01M7 17h.01"/>', o),
    workflow:    o => wrap('<rect x="3" y="3" width="6" height="6" rx="1"/><rect x="15" y="3" width="6" height="6" rx="1"/><rect x="9" y="15" width="6" height="6" rx="1"/><path d="M6 9v3h12v-3"/><path d="M12 12v3"/>', o),
    info:        o => wrap('<circle cx="12" cy="12" r="9"/><path d="M12 8h.01"/><path d="M11 12h1v4h1"/>', o),
    check:       o => wrap('<path d="m5 12 5 5L20 7"/>', o),
    x:           o => wrap('<path d="M6 6 18 18M18 6 6 18"/>', o),
    drag:        o => wrap('<circle cx="9" cy="6" r="1.5"/><circle cx="9" cy="12" r="1.5"/><circle cx="9" cy="18" r="1.5"/><circle cx="15" cy="6" r="1.5"/><circle cx="15" cy="12" r="1.5"/><circle cx="15" cy="18" r="1.5"/>', o),
    db:          o => wrap('<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>', o),
    augment:     o => wrap('<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><path d="M10 7h4M14 17h-4"/><path d="m17 5 3 3-3 3"/><path d="m7 19-3-3 3-3"/>', o),
    quant:       o => wrap('<path d="M4 20V8M9 20V4M14 20v-8M19 20v-4"/>', o),
    eraser:      o => wrap('<path d="m18 13-7 7H5l-2-2 7-7"/><path d="m9 11 5-5 6 6-5 5"/>', o),
    fork:        o => wrap('<circle cx="6" cy="5" r="2"/><circle cx="18" cy="5" r="2"/><circle cx="12" cy="19" r="2"/><path d="M6 7v3a3 3 0 0 0 3 3h6a3 3 0 0 0 3-3V7"/><path d="M12 13v4"/>', o),
    expand:      o => wrap('<path d="M3 9V3h6M21 9V3h-6M3 15v6h6M21 15v6h-6"/>', o),
    pin:         o => wrap('<path d="M12 2 9 9l-6 1 5 4-2 7 6-4 6 4-2-7 5-4-6-1Z"/>', o),
  };
  window.ICONS = I;
})();
