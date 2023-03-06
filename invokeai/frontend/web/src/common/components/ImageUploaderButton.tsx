import { Flex, Heading, Icon } from '@chakra-ui/react';
import { ImageUploaderTriggerContext } from 'app/contexts/ImageUploaderTriggerContext';
import { useContext } from 'react';
import { FaUpload } from 'react-icons/fa';

type ImageUploaderButtonProps = {
  styleClass?: string;
};

const ImageUploaderButton = (props: ImageUploaderButtonProps) => {
  const { styleClass } = props;
  const open = useContext(ImageUploaderTriggerContext);

  const handleClickUpload = () => {
    open && open();
  };

  return (
    <Flex
      sx={{
        width: '100%',
        height: '100%',
        alignItems: 'center',
        justifyContent: 'center',
      }}
      className={styleClass}
    >
      <Flex
        onClick={handleClickUpload}
        sx={{
          display: 'flex',
          flexDirection: 'column',
          rowGap: 8,
          p: 8,
          borderRadius: 'base',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          cursor: 'pointer',
          color: 'base.600',
          bg: 'base.800',
          _hover: {
            bg: 'base.700',
          },
        }}
      >
        <Icon as={FaUpload} boxSize={24} />
        <Heading size="md">Click or Drag and Drop</Heading>
      </Flex>
    </Flex>
  );
};

export default ImageUploaderButton;
