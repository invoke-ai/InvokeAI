// import { Collapse, Flex, useDisclosure } from '@chakra-ui/react';
// import { memo, useState } from 'react';
// import CannyProcessor from './processors/CannyProcessor';
// import { ImageDTO } from 'services/api';
// import IAICustomSelect from 'common/components/IAICustomSelect';
// import {
//   CONTROLNET_PROCESSORS,
//   ControlNetProcessor,
// } from '../store/controlNetSlice';
// import IAISwitch from 'common/components/IAISwitch';

// export type ControlNetProcessorProps = {
//   controlNetId: string;
//   controlImage: ImageDTO | null;
//   processedControlImage: ImageDTO | null;
//   type: ControlNetProcessor;
// };

// const ProcessorComponent = (props: ControlNetProcessorProps) => {
//   const { type } = props;
//   if (type === 'canny') {
//     return <CannyProcessor {...props} />;
//   }
//   return null;
// };

// type ControlNetProcessorCollapseProps = {
//   isOpen: boolean;
//   controlNetId: string;
//   controlImage: ImageDTO | null;
//   processedControlImage: ImageDTO | null;
// };
// const ControlNetProcessorCollapse = (
//   props: ControlNetProcessorCollapseProps
// ) => {
//   const { isOpen, controlImage, controlNetId, processedControlImage } = props;

//   const [processorType, setProcessorType] =
//     useState<ControlNetProcessor>('canny');

//   const handleProcessorTypeChanged = (type: string | null | undefined) => {
//     setProcessorType(type as ControlNetProcessor);
//   };

//   return (
//     <Flex
//       sx={{
//         gap: 2,
//         p: 4,
//         mt: 2,
//         bg: 'base.850',
//         borderRadius: 'base',
//         flexDirection: 'column',
//       }}
//     >
//       <IAICustomSelect
//         label="Processor"
//         items={CONTROLNET_PROCESSORS}
//         selectedItem={processorType}
//         setSelectedItem={handleProcessorTypeChanged}
//       />
//       {controlImage && (
//         <ProcessorComponent
//           controlNetId={controlNetId}
//           controlImage={controlImage}
//           processedControlImage={processedControlImage}
//           type={processorType}
//         />
//       )}
//     </Flex>
//   );
// };

// export default memo(ControlNetProcessorCollapse);

export default {};
