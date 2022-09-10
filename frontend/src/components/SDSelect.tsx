import {
    FormControl,
    FormLabel,
    HStack,
    Select,
    SelectProps,
    Text,
} from '@chakra-ui/react';

interface Props extends SelectProps {
    label: string;
    validValues: Array<number | string>;
}

const SDSelect = (props: Props) => {
    const {
        label,
        isDisabled,
        validValues,
        size = 'sm',
        fontSize = 'sm',
        marginInlineEnd = 0,
        marginBottom = 1,
        whiteSpace = 'nowrap',
        ...rest
    } = props;
    return (
        <FormControl isDisabled={isDisabled}>
            <HStack>
                <FormLabel
                    marginInlineEnd={marginInlineEnd}
                    marginBottom={marginBottom}
                >
                    <Text fontSize={fontSize} whiteSpace={whiteSpace}>
                        {label}
                    </Text>
                </FormLabel>
                <Select fontSize={fontSize} size={size} {...rest}>
                    {validValues.map((val) => (
                        <option key={val} value={val}>
                            {val}
                        </option>
                    ))}
                </Select>
            </HStack>
        </FormControl>
    );
};

export default SDSelect;
