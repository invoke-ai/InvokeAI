import InvokeButton from './InvokeButton';
import CancelButton from './CancelButton';
import LoopbackButton from './Loopback';

/**
 * Buttons to start and cancel image generation.
 */
const ProcessButtons = () => {
  return (
    <div className="process-buttons">
      <InvokeButton />
      <LoopbackButton />
      <CancelButton />
    </div>
  );
};

export default ProcessButtons;
