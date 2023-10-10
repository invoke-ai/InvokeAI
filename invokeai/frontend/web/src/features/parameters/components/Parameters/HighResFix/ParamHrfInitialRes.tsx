import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import {
  setHrfHeight,
  setHrfWidth,
} from 'features/parameters/store/generationSlice';
import {
  HeightParam,
  WidthParam,
  ResolutionParam,
} from 'features/parameters/types/parameterSchemas';
import { useCallback, useState, useEffect } from 'react';

// Finds the top three largest resolutions that maintain the same aspect ratio
// but are divisible by eight.
function findLargestResolutionsDivisibleByEight(
  width: WidthParam,
  height: HeightParam
): ResolutionParam[] {
  const aspectRatio = width / height;
  const results: ResolutionParam[] = [];
  // Find the largest dimension divisible by 8 that's less than the original
  let newWidth = Math.floor((width - 1) / 8) * 8;
  let newHeight = Math.round(newWidth / aspectRatio);

  while (results.length < 3) {
    // Stop if dimensions go below a reasonable limit
    if (newWidth < 8 || newHeight < 8) {
      break;
    }

    // Ensure both dimensions are divisible by 8
    if (newWidth % 8 === 0 && newHeight % 8 === 0) {
      results.push([newWidth, newHeight]);
    }

    // Decrement dimensions while maintaining aspect ratio
    newWidth -= 8;
    newHeight = Math.round(newWidth / aspectRatio);
  }
  return results;
}

// Contains the code for the dropdown specifying the
// initial resolution of the High Resolution Fix feature.
export default function ParamHrfInitialRes() {
  const { height, width, hrfEnabled } = useAppSelector((state: RootState) => ({
    height: state.generation.height,
    width: state.generation.width,
    hrfEnabled: state.generation.hrfEnabled,
  }));
  const dispatch = useAppDispatch();

  // Local state to hold dropdown options
  const [dropdownOptions, setDropdownOptions] = useState<string[]>([]);

  // Update dropdown options when width, height, or hrfEnabled change
  useEffect(() => {
    if (hrfEnabled) {
      const resolutions = findLargestResolutionsDivisibleByEight(width, height);
      const options = resolutions.map(([w, h]) => `${w},${h}`);
      setDropdownOptions(options);

      if (options.length > 0) {
        const [initialWidth, initialHeight] = options[0]!
          .split(',')
          .map(Number);
        dispatch(setHrfWidth(initialWidth as WidthParam));
        dispatch(setHrfHeight(initialHeight as HeightParam));
      }
    }
  }, [width, height, hrfEnabled, dispatch]);

  const handleChange = useCallback(
    (option: string | null) => {
      if (!option) {
        return;
      }
      const [width, height] = option.split(',').map(Number);
      dispatch(setHrfWidth(width as WidthParam));
      dispatch(setHrfHeight(height as HeightParam));
    },
    [dispatch]
  );

  return (
    <IAIMantineSelect
      label="Initial Resolution"
      value=""
      data={dropdownOptions}
      onChange={handleChange}
    />
  );
}
