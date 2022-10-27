import TextToImagePanel from './TextToImagePanel';
import CurrentImageDisplay from '../../gallery/CurrentImageDisplay';
import InvokeWorkarea from '../InvokeWorkarea';

export default function TextToImageWorkarea() {
  return (
    <InvokeWorkarea
      optionsPanel={<TextToImagePanel />}
      className="txt-to-image-workarea-container"
    >
      <CurrentImageDisplay />
    </InvokeWorkarea>
  );
}
