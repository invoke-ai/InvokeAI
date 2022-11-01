import InpaintingBrushControl from './InpaintingControls/InpaintingBrushControl';
import InpaintingEraserControl from './InpaintingControls/InpaintingEraserControl';
import InpaintingMaskControl from './InpaintingControls/InpaintingMaskControl';
import InpaintingUndoControl from './InpaintingControls/InpaintingUndoControl';
import InpaintingRedoControl from './InpaintingControls/InpaintingRedoControl';
import InpaintingClearImageControl from './InpaintingControls/InpaintingClearImageControl';
import InpaintingSplitLayoutControl from './InpaintingControls/InpaintingSplitLayoutControl';

const InpaintingControls = () => {
  // const { shouldShowMask, activeTabName } = useAppSelector(
  //   inpaintingControlsSelector
  // );

  // const dispatch = useAppDispatch();

  /**
   * Hotkeys
   */

  // Toggle lock bounding box
  // useHotkeys(
  //   'shift+q',
  //   (e: KeyboardEvent) => {
  //     e.preventDefault();
  //     dispatch(toggleShouldLockBoundingBox());
  //   },
  //   {
  //     enabled: activeTabName === 'inpainting' && shouldShowMask,
  //   },
  //   [activeTabName, shouldShowMask]
  // );

  return (
    <div className="inpainting-settings">
      <div className="inpainting-buttons-group">
        <InpaintingBrushControl />
        <InpaintingEraserControl />
      </div>
      <div className="inpainting-buttons-group">
        <InpaintingMaskControl />
      </div>
      <div className="inpainting-buttons-group">
        <InpaintingUndoControl />
        <InpaintingRedoControl />
      </div>

      <div className="inpainting-buttons-group">
        <InpaintingClearImageControl />
      </div>
      <InpaintingSplitLayoutControl />
    </div>
  );
};

export default InpaintingControls;
