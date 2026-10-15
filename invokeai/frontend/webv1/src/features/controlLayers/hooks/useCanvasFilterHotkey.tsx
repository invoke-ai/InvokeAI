import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useEntityFilter } from 'features/controlLayers/hooks/useEntityFilter';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';

export const useCanvasFilterHotkey = () => {
  useAssertSingleton('useCanvasFilterHotkey');
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const filter = useEntityFilter(selectedEntityIdentifier);

  useRegisteredHotkeys({
    id: 'filterSelected',
    category: 'canvas',
    callback: filter.start,
    options: { enabled: !filter.isDisabled, preventDefault: true },
    dependencies: [filter],
  });
};
