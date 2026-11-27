import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useEntityTransform } from 'features/controlLayers/hooks/useEntityTransform';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';

export const useCanvasTransformHotkey = () => {
  useAssertSingleton('useCanvasTransformHotkey');
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const transform = useEntityTransform(selectedEntityIdentifier);

  useRegisteredHotkeys({
    id: 'transformSelected',
    category: 'canvas',
    callback: transform.start,
    options: { enabled: !transform.isDisabled, preventDefault: true },
    dependencies: [transform],
  });
};
