import { useCatalog } from '../state/CatalogContext';
import { useSession } from '../state/SessionContext';

const IMAGE_HINT = /mnist|image|digit|pixel|conv|cifar/i;

/**
 * Work out whether the current task should be shown as pixels rather than
 * numbers, and at what dimensions.
 *
 * Priority: explicit user override → selected dataset metadata → a square input
 * vector on a function whose key looks image-shaped.
 */
export function resolveIoShape({ func = null, dataset = null, config = null, viz = null } = {}) {
  const inputs = Number(func?.inputs ?? config?.inputs ?? 0);
  const outputs = Number(func?.outputs ?? config?.outputs ?? 0);

  let input = null;
  if (viz?.imageInOverride) input = viz.imageInOverride;
  else if (dataset?.width && dataset?.height) {
    input = [dataset.width, dataset.height, dataset.channels || 1];
  } else {
    const side = Math.round(Math.sqrt(inputs));
    const looksImagey = IMAGE_HINT.test(func?.key || '') || IMAGE_HINT.test(func?.label || '');
    if (inputs >= 16 && side * side === inputs && looksImagey) input = [side, side, 1];
  }

  let output = null;
  if (viz?.imageOutOverride) output = viz.imageOutOverride;
  else if (input && outputs >= 4) {
    const side = Math.round(Math.sqrt(outputs));
    if (side * side === outputs) output = [side, side, 1];
  }

  const forced = Boolean(viz?.showImageIO);
  if (forced && !input && inputs >= 4) {
    const side = Math.max(2, Math.round(Math.sqrt(inputs)));
    input = [side, side, 1];
  }

  return {
    input,
    output,
    isImage: Boolean(input || output),
    inputs,
    outputs,
  };
}

/** React binding for resolveIoShape against the live session + catalogue. */
export function useIoShape() {
  const func = useSession((s) => s.snapshot?.func);
  const config = useSession((s) => s.config);
  const viz = useSession((s) => s.viz);
  const catalog = useCatalog();
  const dataset = config.dsId ? catalog.datasetById?.[String(config.dsId)] || null : null;
  return resolveIoShape({ func, dataset, config, viz });
}
