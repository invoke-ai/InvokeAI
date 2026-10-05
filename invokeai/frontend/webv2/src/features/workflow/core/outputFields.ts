import type { FieldOutputTemplate } from './types';

export interface OutputFieldNamesByScope {
  all: string[];
  unscoped: string[];
  iteration: string[];
  final: string[];
}

export type OutputFieldRow = { type: 'field'; fieldName: string } | { type: 'header'; scope: 'iteration' | 'final' };

export const getOutputFieldNamesByScope = (fields: FieldOutputTemplate[]): OutputFieldNamesByScope => {
  const all = fields.filter((field) => !field.uiHidden).map((field) => field.name);
  const fieldsByName = new Map(fields.map((field) => [field.name, field]));

  return {
    all,
    final: all.filter((name) => fieldsByName.get(name)?.outputScope === 'final'),
    iteration: all.filter((name) => fieldsByName.get(name)?.outputScope === 'iteration'),
    unscoped: all.filter((name) => !fieldsByName.get(name)?.outputScope),
  };
};

export const getOutputFieldRows = (fieldNames: OutputFieldNamesByScope): OutputFieldRow[] => {
  if (fieldNames.iteration.length === 0 && fieldNames.final.length === 0) {
    return fieldNames.all.map((fieldName) => ({ fieldName, type: 'field' as const }));
  }

  const rows: OutputFieldRow[] = fieldNames.unscoped.map((fieldName) => ({ fieldName, type: 'field' as const }));

  if (fieldNames.iteration.length > 0) {
    rows.push({ scope: 'iteration', type: 'header' });
    rows.push(...fieldNames.iteration.map((fieldName) => ({ fieldName, type: 'field' as const })));
  }

  if (fieldNames.final.length > 0) {
    rows.push({ scope: 'final', type: 'header' });
    rows.push(...fieldNames.final.map((fieldName) => ({ fieldName, type: 'field' as const })));
  }

  return rows;
};

const OUTPUT_VALUE_MAX_LENGTH = 40;
const NAMED_OBJECT_KEYS = ['image_name', 'video_name', 'latents_name', 'tensor_name', 'name', 'key', 'board_id'];

/** Compact label for one output field's runtime value; `null` when the result carries no such field. */
export const formatOutputFieldValue = (result: unknown, fieldName: string): { full: string; short: string } | null => {
  if (typeof result !== 'object' || result === null || !(fieldName in result)) {
    return null;
  }

  const value = (result as Record<string, unknown>)[fieldName];
  const full = formatOutputValue(value);

  return {
    full,
    short: full.length > OUTPUT_VALUE_MAX_LENGTH ? `${full.slice(0, OUTPUT_VALUE_MAX_LENGTH - 1)}…` : full,
  };
};

const formatOutputValue = (value: unknown): string => {
  if (value === null || value === undefined) {
    return '—';
  }

  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(4)));
  }

  if (typeof value === 'string' || typeof value === 'boolean') {
    return String(value);
  }

  if (Array.isArray(value)) {
    return `${value.length} item${value.length === 1 ? '' : 's'}`;
  }

  if (typeof value === 'object') {
    const record = value as Record<string, unknown>;

    for (const key of NAMED_OBJECT_KEYS) {
      if (typeof record[key] === 'string') {
        return record[key];
      }
    }

    if (['r', 'g', 'b', 'a'].every((channel) => typeof record[channel] === 'number')) {
      return `rgba(${record.r}, ${record.g}, ${record.b}, ${record.a})`;
    }

    return `{${Object.keys(record).length} fields}`;
  }

  return String(value);
};
