import { As, Badge, Flex } from '@chakra-ui/react';
import { TypesafeDroppableData } from 'app/components/ImageDnd/typesafeDnd';
import IAIDroppable from 'common/components/IAIDroppable';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';

type GenericBoardProps = {
  droppableData: TypesafeDroppableData;
  onClick: () => void;
  isSelected: boolean;
  icon: As;
  label: string;
  badgeCount?: number;
};

const GenericBoard = (props: GenericBoardProps) => {
  const { droppableData, onClick, isSelected, icon, label, badgeCount } = props;

  return (
    <Flex
      sx={{
        flexDir: 'column',
        justifyContent: 'space-between',
        alignItems: 'center',
        cursor: 'pointer',
        w: 'full',
        h: 'full',
        borderRadius: 'base',
      }}
    >
      <Flex
        onClick={onClick}
        sx={{
          position: 'relative',
          justifyContent: 'center',
          alignItems: 'center',
          borderRadius: 'base',
          w: 'full',
          aspectRatio: '1/1',
          overflow: 'hidden',
          shadow: isSelected ? 'selected.light' : undefined,
          _dark: { shadow: isSelected ? 'selected.dark' : undefined },
          flexShrink: 0,
        }}
      >
        <IAINoContentFallback
          boxSize={8}
          icon={icon}
          sx={{
            border: '2px solid var(--invokeai-colors-base-200)',
            _dark: { border: '2px solid var(--invokeai-colors-base-800)' },
          }}
        />
        <Flex
          sx={{
            position: 'absolute',
            insetInlineEnd: 0,
            top: 0,
            p: 1,
          }}
        >
          {badgeCount !== undefined && (
            <Badge variant="solid">{badgeCount}</Badge>
          )}
        </Flex>
        <IAIDroppable data={droppableData} />
      </Flex>
      <Flex
        sx={{
          h: 'full',
          alignItems: 'center',
          fontWeight: isSelected ? 600 : undefined,
          fontSize: 'xs',
          color: isSelected ? 'base.900' : 'base.700',
          _dark: { color: isSelected ? 'base.50' : 'base.200' },
        }}
      >
        {label}
      </Flex>
    </Flex>
  );
};

export default GenericBoard;
