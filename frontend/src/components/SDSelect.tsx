import { FormControl, FormLabel, HStack, Select, Text } from '@chakra-ui/react';
import { ChangeEventHandler } from 'react';

type Props = {
    label: string;
    value: string | number;
    onChange: ChangeEventHandler<HTMLSelectElement>;
    validValues: Array<string | number>;
    isDisabled?: boolean;
};

const SDSelect = ({
    label,
    value,
    onChange,
    validValues,
    isDisabled = false,
}: Props) => {
    return (
        <FormControl isDisabled={isDisabled}>
            <HStack>
                <FormLabel>
                    <Text fontSize={'sm'} whiteSpace='nowrap'>
                        {label}
                    </Text>
                </FormLabel>
                <Select
                    fontSize={'sm'}
                    size={'sm'}
                    onChange={onChange}
                    value={value}
                >
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
