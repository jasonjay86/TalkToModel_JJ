import pickletools

with open("austrailian_modelRF.pkl", "rb") as f:
    pickle = f.read()
    output = pickletools.genops(pickle)
    opcodes = []
    for opcode in output:
        opcodes.append(opcode[0])
    print(opcodes[0].name)
    print(opcodes[-1].name)