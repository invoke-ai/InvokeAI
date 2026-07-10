import { describe, expect, it } from 'vitest';

import * as layerGroupSectionModule from './LayerGroupSection.tsx?raw';
import * as layerListItemModule from './LayerListItem.tsx?raw';

const layerGroupSectionSource = layerGroupSectionModule.default;
const layerListItemSource = layerListItemModule.default;

describe('layer drag-and-drop wiring', () => {
  it('registers pointer dragging without a keyboard sensor', () => {
    expect(layerGroupSectionSource).toContain('useSensor(PointerSensor, POINTER_SENSOR_OPTIONS)');
    expect(layerGroupSectionSource).not.toContain('KeyboardSensor');
    expect(layerGroupSectionSource).not.toContain('sortableKeyboardCoordinates');
  });

  it('keeps pointer listeners on the row without keyboard-interactive sortable attributes', () => {
    expect(layerListItemSource).toContain('{...listeners}');
    expect(layerListItemSource).not.toContain('{...attributes}');
  });
});
