import type { FieldType } from 'features/nodes/types/field';
import {
  isLoRAFieldCollectionFieldType,
  isLoRAFieldCollectionInputInstance,
  isLoRAFieldCollectionInputTemplate,
  zLoRAFieldCollectionValue,
} from 'features/nodes/types/field';
import { describe, expect, it } from 'vitest';

const loraValue = {
  lora: {
    key: 'some-key',
    hash: 'some-hash',
    name: 'My LoRA',
    base: 'sdxl',
    type: 'lora',
  },
  weight: 0.75,
};

describe('LoRAField collection', () => {
  describe('isLoRAFieldCollectionFieldType', () => {
    it.each([
      [{ name: 'LoRAField', cardinality: 'COLLECTION', batch: false }, true],
      [{ name: 'LoRAField', cardinality: 'SINGLE_OR_COLLECTION', batch: false }, true],
      // A bare SINGLE LoRAField (e.g. the Select LoRA output) is not the editable collection type
      [{ name: 'LoRAField', cardinality: 'SINGLE', batch: false }, false],
      [{ name: 'ImageField', cardinality: 'COLLECTION', batch: false }, false],
    ] satisfies [FieldType, boolean][])('%o -> %s', (fieldType, expected) => {
      expect(isLoRAFieldCollectionFieldType(fieldType)).toBe(expected);
    });
  });

  describe('zLoRAFieldCollectionValue', () => {
    it('accepts a list of lora/weight pairs', () => {
      expect(zLoRAFieldCollectionValue.safeParse([loraValue]).success).toBe(true);
    });
    it('accepts an empty list (an empty collection is valid and lets the graph fire)', () => {
      expect(zLoRAFieldCollectionValue.safeParse([]).success).toBe(true);
    });
    it('accepts undefined', () => {
      expect(zLoRAFieldCollectionValue.safeParse(undefined).success).toBe(true);
    });
    it('rejects a non-numeric weight', () => {
      expect(zLoRAFieldCollectionValue.safeParse([{ ...loraValue, weight: 'heavy' }]).success).toBe(false);
    });
    it('rejects an item missing its lora identifier', () => {
      expect(zLoRAFieldCollectionValue.safeParse([{ weight: 1 }]).success).toBe(false);
    });
  });

  describe('input instance / template guards', () => {
    it('recognizes a valid collection input instance', () => {
      const instance = { name: 'loras', label: '', description: '', value: [loraValue] };
      expect(isLoRAFieldCollectionInputInstance(instance)).toBe(true);
    });

    it('recognizes a collection input template (SINGLE_OR_COLLECTION)', () => {
      const template = {
        name: 'loras',
        title: 'LoRAs',
        description: '',
        ui_hidden: false,
        fieldKind: 'input',
        input: 'any',
        required: false,
        type: { name: 'LoRAField', cardinality: 'SINGLE_OR_COLLECTION', batch: false },
        default: [],
      };
      expect(isLoRAFieldCollectionInputTemplate(template)).toBe(true);
    });
  });
});
