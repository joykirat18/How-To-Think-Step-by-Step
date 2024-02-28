TemplateCOT_fictional = """### Input:
Gorpus is twimpus. Alex is rompus. Rompus is gorpus. Gorpus is small. Rompus is mean. True or false: Alex is small. Let us think step by step.
### Response:
Alex is rompus. Rompus is gorpus. Alex is gorpus. Gorpus is small. Alex is small. True

### Input:
Gorpuses are discordant. Max is zumpus. Zumpus is shampor. Zumpus is gorpus. Gorpus is earthy. True or false: Max is small. Let us think step by step. 
### Response:
Max is zumpus. Zumpus is gorpus. Max is gorpus. Gorpuses are discordant. Max is discordant. False

### Input:
Borpin are wumpus. Wumpuses are angry. Wumpus is jempor. Sally is lempus. Lempus is wumpus. True or false: Sally is floral. Let us think step by step. 
### Response:
Sally is lempus. Lempus is wumpus. Sally is wumpus. Wumpuses are angry. Sally is angry. False

### Input:
Gorpus is jelgit. Yumpuses are loud. Gorpus is yumpus. Yumpus is orange. Rex is gorpus. True or false: Rex is loud. Let us think step by step.
### Response:
Rex is gorpus. Gorpus is yumpus. Rex is yumpus. Yumpuses are loud. Rex is loud. True

### Input:
Lempus is tumpus. Max is lempus. Tumpus is fruity. Tumpus is bright. Lempus is dropant. True or false: Max is bright. Let us think step by step.
### Response:
Max is lempus. Lempus is tumpus. Max is tumpus. Tumpus is bright. Max is bright. True

### Input:
Sterpuses are dull. Impus is medium. Impuses are sterpuses. Wren is impus. Sterpus is daumpin. True or false: Wren is melodic. Let us think step by step.
### Response:
Wren is impus. Impuses are sterpuses. Wren is sterpus. Sterpuses are dull. Wren is dull. False

### Input:
{}
### Response:
"""

TemplateCOT_false = """### Input:
Cats are cows. Spider are vertebrate. Rex is cat. Cows are large. Cows are large. True or false: Rex is large. Let us think step by step.
### Response:
Rex is cat. Cats are cows. Rex is cow. Cows are large. Rex is large. True

### Input:
Vertebrates are cows. Sheep is bitter. Cows are sheep. Fae is cow. Crustaceans are large. True or false: Fae is red. Let us think step by step.
### Response:
Fae is cow. Cows are sheep. Fae is sheep. Sheep is bitter. Fae is bitter. False

### Input:
Dogs are mammals. Mammals are snowy. Lepidopterans are crustaceans. Alex is dog. Mammals are arthropods. True or false: Alex is snowy. Let us think step by step.
### Response:
Alex is dog. Dogs are mammals. Alex is mammal. Mammals are snowy. Alex is snowy. True

### Input:
Moths are discordant. Spiders are moths. Sam is mammal. Mammals are snakes. Snakes are spicy. True or false: Sam is spicy. Let us think step by step.
### Response:
Sam is mammal. Mammals are snakes. Sam is snake. Snakes are spicy. Sam is spicy. True

### Input:
Dogs are cows. Max is dog. Butterflies are spiders. Cows are dull. Spiders are floral. True or false: Max is luminous. Let us think step by step.
### Response:
Max is dog. Dogs are cows. Max is cow. Cows are dull. Max is dull. False

### Input:
Dog is feisty. Feline is small. Sheep are melodic. Sam is mammal. Mammals are dogs. True or false: Sam is feisty. Let us think step by step.
### Response:
Sam is mammal. Mammals are dogs. Sam is dog. Dog is feisty. Sam is feisty. True

### Input:
{}
### Response:
"""


TemplateCOT_generic = """### Input:
Carnivores are fruity. Felines are carnivores. Wren is feline. True or false: Wren is fruity. Let us think step by step.
### Response:
Wren is feline. Felines are carnivores. Wren is carnivore. Carnivores are fruity. Wren is fruity. True

### Input:
Moths are small. Spiders are moths. Wren is spider. True or false: Wren is overcast. Let us think step by step.
### Response:
Wren is spider. Spiders are moths. Wren is moth. Moths are small. Wren is small. False

### Input:
Invertebrates are dull. Lepidopterans are invertebrates. Sally is lepidopteran. True or false: Sally is metallic. Let us think step by step.
### Response:
Sally is lepidopteran. Lepidopterans are invertebrates. Sally is invertebrate. Invertebrates are dull. Sally is dull. False

### Input:
Animals are aggressive. Ants are animals. Stella is ant. True or false: Stella is aggressive. Let us think step by step.
### Response:
Stella is ant. Ants are animals. Stella is animal. Animals are aggressive. Stella is aggressive. True

### Input:
{}
### Response:
"""

TemplateCOT_3Hop_fictional = """### Input:
Tumpus is lempus. Lorpuses are small. Gorpus is twimpus. Alex is rompus. Rex is tumpus. Lempus is lorpus. True or false: Rex is muffled. Let us think step by step.
### Response:
Rex is tumpus. Tumpus is lempus. Rex is lempus. Lempus is lorpus. Rex is lorpus. Lorpuses are small. Rex is small. False

### Input:
Yumpus is small. Grimpuses are yumpuses. Shumpus is grimpus. Wren is zumpus. Max is shumpus. Gorpuses are discordant. True or false: Max is small. Let us think step by step.
### Response:
Max is shumpus. Shumpus is grimpus. Max is grimpus. Grimpuses are yumpuses. Max is yumpus. Yumpus is small. Max is small. True

### Input:
Impus is medium. Sterpuses are dull. Brimpuses are fast. Max is jompus. Jompuses are tumpuses. Tumpuses are brimpuses. True or false: Max is fast. Let us think step by step.
### Response:
Max is jompus. Jompuses are tumpuses. Max is tumpus. Tumpuses are brimpuses. Max is brimpus. Brimpuses are fast. Max is fast. True

### Input:
Impus is medium. Sterpuses are dull. Brimpuses are fast. Max is jompus. Jompuses are tumpuses. Tumpuses are brimpuses. True or false: Max is fast. Let us think step by step.
### Response:
Rex is yumpus. Yumpus is wumpus. Rex is wumpus. Wumpuses are zumpuses. Rex is zumpus. Zumpus is bright. Rex is bright. False

### Input:
Yumpuses are loud. Wren is zumpus. Brimpuses are lempuses. Zumpus is brimpus. Lempuses are mean. Gorpus is jelgit. True or false: Wren is bitter. Let us think step by step.
### Response:
Wren is zumpus. Zumpus is brimpus. Wren is brimpus. Brimpuses are lempuses. Wren is lempus. Lempuses are mean. Wren is mean. False

### Input:
Sally is numpus. Lempuses are grimpuses. Gorpus is tumpus. Grimpus is mean. Numpuses are lempuses. Max is lempus. True or false: Sally is mean. Let us think step by step.
### Response:
Sally is numpus. Numpuses are lempuses. Sally is lempus. Lempuses are grimpuses. Sally is grimpus. Grimpus is mean. Sally is mean. True

### Input:
{}
### Response:
"""

TemplateCOT_3Hop_fictional_1step = """### Input:
Tumpus is lempus. Lorpuses are small. Gorpus is twimpus. Alex is rompus. Rex is tumpus. Lempus is lorpus. True or false: Rex is muffled. Let us think step by step.
### Response:
Rex is tumpus. Tumpus is lempus. Lempus is lorpus. Rex is lorpus. Lorpuses are small. Rex is small. False

### Input:
Yumpus is small. Grimpuses are yumpuses. Shumpus is grimpus. Wren is zumpus. Max is shumpus. Gorpuses are discordant. True or false: Max is small. Let us think step by step.
### Response:
Max is shumpus. Shumpus is grimpus. Grimpuses are yumpuses. Max is yumpus. Yumpus is small. Max is small. True

### Input:
Impus is medium. Sterpuses are dull. Brimpuses are fast. Max is jompus. Jompuses are tumpuses. Tumpuses are brimpuses. True or false: Max is fast. Let us think step by step.
### Response:
Max is jompus. Jompuses are tumpuses. Tumpuses are brimpuses. Max is brimpus. Brimpuses are fast. Max is fast. True

### Input:
Wumpuses are angry. Borpin are wumpus. Rex is yumpus. Wumpuses are zumpuses. Zumpus is bright. Yumpus is wumpus. True or false: Rex is happy. Let us think step by step.
### Response:
Rex is yumpus. Yumpus is wumpus. Wumpuses are zumpuses. Rex is zumpus. Zumpus is bright. Rex is bright. False

### Input:
Yumpuses are loud. Wren is zumpus. Brimpuses are lempuses. Zumpus is brimpus. Lempuses are mean. Gorpus is jelgit. True or false: Wren is bitter. Let us think step by step.
### Response:
Wren is zumpus. Zumpus is brimpus. Brimpuses are lempuses. Wren is lempus. Lempuses are mean. Wren is mean. False

### Input:
Sally is numpus. Lempuses are grimpuses. Gorpus is tumpus. Grimpus is mean. Numpuses are lempuses. Max is lempus. True or false: Sally is mean. Let us think step by step.
### Response:
Sally is numpus. Numpuses are lempuses. Lempuses are grimpuses. Sally is grimpus. Grimpus is mean. Sally is mean. True

### Input:
{}
### Response:
"""

TemplateCOT_3Hop_fictional_2step = """### Input:
Tumpus is lempus. Lorpuses are small. Gorpus is twimpus. Alex is rompus. Rex is tumpus. Lempus is lorpus. True or false: Rex is muffled. Let us think step by step.
### Response:
Rex is tumpus. Tumpus is lempus. Lempus is lorpus. Lorpuses are small. Rex is small. False

### Input:
Yumpus is small. Grimpuses are yumpuses. Shumpus is grimpus. Wren is zumpus. Max is shumpus. Gorpuses are discordant. True or false: Max is small. Let us think step by step.
### Response:
Max is shumpus. Shumpus is grimpus. Grimpuses are yumpuses. Yumpus is small. Max is small. True

### Input:
Impus is medium. Sterpuses are dull. Brimpuses are fast. Max is jompus. Jompuses are tumpuses. Tumpuses are brimpuses. True or false: Max is fast. Let us think step by step.
### Response:
Max is jompus. Jompuses are tumpuses. Tumpuses are brimpuses. Brimpuses are fast. Max is fast. True

### Input:
Wumpuses are angry. Borpin are wumpus. Rex is yumpus. Wumpuses are zumpuses. Zumpus is bright. Yumpus is wumpus. True or false: Rex is happy. Let us think step by step.
### Response:
Rex is yumpus. Yumpus is wumpus. Wumpuses are zumpuses. Zumpus is bright. Rex is bright. False

### Input:
Yumpuses are loud. Wren is zumpus. Brimpuses are lempuses. Zumpus is brimpus. Lempuses are mean. Gorpus is jelgit. True or false: Wren is bitter. Let us think step by step.
### Response:
Wren is zumpus. Zumpus is brimpus. Brimpuses are lempuses. Lempuses are mean. Wren is mean. False

### Input:
Sally is numpus. Lempuses are grimpuses. Gorpus is tumpus. Grimpus is mean. Numpuses are lempuses. Max is lempus. True or false: Sally is mean. Let us think step by step.
### Response:
Sally is numpus. Numpuses are lempuses. Lempuses are grimpuses. Grimpus is mean. Sally is mean. True

### Input:
{}
### Response:
"""

TemplateCOT_3Hop_false = """### Input:
Vertebrate is cat. Fae is cow. Crustaceans are large. Snake is metallic. Stella is vertebrate. Cats are snakes. True or false: Stella is red. Let us think step by step.
### Response:
Stella is vertebrate. Vertebrate is cat. Stella is cat. Cats are snakes. Stella is snake. Snake is metallic. Stella is metallic. False

### Input:
Sheep is bitter. Cats are snakes. Carnivores are amenable. Sam is cat. Vertebrates are cows. Snake is carnivore. True or false: Sam is amenable. Let us think step by step.
### Response:
Sam is cat. Cats are snakes. Sam is snake. Snake is carnivore. Sam is carnivore. Carnivores are amenable. Sam is amenable. True

### Input:
Mammals are snowy. Carnivore is fast. Cow is feline. Feline is carnivore. Dogs are mammals. Stella is cow. True or false: Stella is opaque. Let us think step by step.
### Response:
Stella is cow. Cow is feline. Stella is feline. Feline is carnivore. Stella is carnivore. Carnivore is fast. Stella is fast. False

### Input:
Spiders are moths. Sheep is loud. Fae is mammal. Animals are sheep. Mammal is animal. Moths are discordant. True or false: Fae is loud. Let us think step by step.
### Response:
Fae is mammal. Mammal is animal. Fae is animal. Animals are sheep. Fae is sheep. Sheep is loud. Fae is loud. True

### Input:
Dogs are cows. Vertebrates are feisty. Sam is mammal. Mammals are cats. Cat is vertebrate. Max is dog. True or false: Sam is sour. Let us think step by step.
### Response:
Sam is mammal. Mammals are cats. Sam is cat. Cat is vertebrate. Sam is vertebrate. Vertebrates are feisty. Sam is feisty. False

### Input:
Dog is feisty. Feline is small. Sam is cow. Snakes are sweet. Mammals are snakes. Cows are mammals. True or false: Sam is sweet. Let us think step by step.
### Response:
Sam is cow. Cows are mammals. Sam is mammal. Mammals are snakes. Sam is snake. Snakes are sweet. Sam is sweet. True

### Input:
{}
### Response:
"""

TemplateCOT_3Hop_false_1step = """### Input:
Vertebrate is cat. Fae is cow. Crustaceans are large. Snake is metallic. Stella is vertebrate. Cats are snakes. True or false: Stella is red. Let us think step by step.
### Response:
Stella is vertebrate. Vertebrate is cat. Cats are snakes. Stella is snake. Snake is metallic. Stella is metallic. False

### Input:
Sheep is bitter. Cats are snakes. Carnivores are amenable. Sam is cat. Vertebrates are cows. Snake is carnivore. True or false: Sam is amenable. Let us think step by step.
### Response:
Sam is cat. Cats are snakes. Snake is carnivore. Sam is carnivore. Carnivores are amenable. Sam is amenable. True

### Input:
Mammals are snowy. Carnivore is fast. Cow is feline. Feline is carnivore. Dogs are mammals. Stella is cow. True or false: Stella is opaque. Let us think step by step.
### Response:
Stella is cow. Cow is feline. Feline is carnivore. Stella is carnivore. Carnivore is fast. Stella is fast. False

### Input:
Spiders are moths. Sheep is loud. Fae is mammal. Animals are sheep. Mammal is animal. Moths are discordant. True or false: Fae is loud. Let us think step by step.
### Response:
Fae is mammal. Mammal is animal. Animals are sheep. Fae is sheep. Sheep is loud. Fae is loud. True

### Input:
Dogs are cows. Vertebrates are feisty. Sam is mammal. Mammals are cats. Cat is vertebrate. Max is dog. True or false: Sam is sour. Let us think step by step.
### Response:
Sam is mammal. Mammals are cats. Cat is vertebrate. Sam is vertebrate. Vertebrates are feisty. Sam is feisty. False

### Input:
Dog is feisty. Feline is small. Sam is cow. Snakes are sweet. Mammals are snakes. Cows are mammals. True or false: Sam is sweet. Let us think step by step.
### Response:
Sam is cow. Cows are mammals. Mammals are snakes. Sam is snake. Snakes are sweet. Sam is sweet. True

### Input:
{}
### Response:
"""

TemplateCOT_3Hop_false_2step = """### Input:
Vertebrate is cat. Fae is cow. Crustaceans are large. Snake is metallic. Stella is vertebrate. Cats are snakes. True or false: Stella is red. Let us think step by step.
### Response:
Stella is vertebrate. Vertebrate is cat. Cats are snakes. Snake is metallic. Stella is metallic. False

### Input:
Sheep is bitter. Cats are snakes. Carnivores are amenable. Sam is cat. Vertebrates are cows. Snake is carnivore. True or false: Sam is amenable. Let us think step by step.
### Response:
Sam is cat. Cats are snakes. Snake is carnivore. Carnivores are amenable. Sam is amenable. True

### Input:
Mammals are snowy. Carnivore is fast. Cow is feline. Feline is carnivore. Dogs are mammals. Stella is cow. True or false: Stella is opaque. Let us think step by step.
### Response:
Stella is cow. Cow is feline. Feline is carnivore. Carnivore is fast. Stella is fast. False

### Input:
Spiders are moths. Sheep is loud. Fae is mammal. Animals are sheep. Mammal is animal. Moths are discordant. True or false: Fae is loud. Let us think step by step.
### Response:
Fae is mammal. Mammal is animal. Animals are sheep. Sheep is loud. Fae is loud. True

### Input:
Dogs are cows. Vertebrates are feisty. Sam is mammal. Mammals are cats. Cat is vertebrate. Max is dog. True or false: Sam is sour. Let us think step by step.
### Response:
Sam is mammal. Mammals are cats. Cat is vertebrate. Vertebrates are feisty. Sam is feisty. False

### Input:
Dog is feisty. Feline is small. Sam is cow. Snakes are sweet. Mammals are snakes. Cows are mammals. True or false: Sam is sweet. Let us think step by step.
### Response:
Sam is cow. Cows are mammals. Mammals are snakes. Snakes are sweet. Sam is sweet. True

### Input:
{}
### Response:
"""

TemplateCOT_4Hop_false = """
### Input:
Animals are cows. Fae is cow. Vertebrates are animals. Cow is kind. Feline is vertebrate. Crustaceans are large. Rex is feline. True or false: Rex is kind. Let us think step by step.
### Response:
Rex is feline. Feline is vertebrate. Rex is vertebrate. Vertebrates are animals. Rex is animal. Animals are cows. Rex is cow. Cow is kind. Rex is kind. True

### Input:
Vertebrates are mammals. Dogs are vertebrates. Vertebrates are cows. Sheep is bitter. Wren is carnivore. Carnivores are dogs. Mammal is spicy. True or false: Wren is temperate. Let us think step by step.
### Response:
Wren is carnivore. Carnivores are dogs. Wren is dog. Dogs are vertebrates. Wren is vertebrate. Vertebrates are mammals. Wren is mammal. Mammal is spicy. Wren is spicy. False

### Input:
Butterflies are animals. Dogs are mammals. Lepidopterans are temperate. Ants are lepidopterans. Rex is butterfly. Animals are ants. Mammals are snowy. True or false: Rex is temperate. Let us think step by step.
### Response:
Rex is butterfly. Butterflies are animals. Rex is animal. Animals are ants. Rex is ant. Ants are lepidopterans. Rex is lepidopteran. Lepidopterans are temperate. Rex is temperate. True

### Input:
Vertebrate is red. Carnivores are mammals. Polly is carnivore. Moths are discordant. Mammals are dogs. Dog is vertebrate. Spiders are moths. True or false: Polly is sweet. Let us think step by step.
### Response:
Polly is carnivore. Carnivores are mammals. Polly is mammal. Mammals are dogs. Polly is dog. Dog is vertebrate. Polly is vertebrate. Vertebrate is red. Polly is red. False

### Input:
Cows are snakes. Dogs are cows. Stella is cow. Max is dog. Carnivore is mammal. Snakes are carnivores. Mammals are amenable. True or false: Stella is amenable. Let us think step by step.
### Response:
Stella is cow. Cows are snakes. Stella is snake. Snakes are carnivores. Stella is carnivore. Carnivore is mammal. Stella is mammal. Mammals are amenable. Stella is amenable. True

### Input:
Sally is insect. Insects are crustaceans. Invertebrate is animal. Feline is small. Crustaceans are invertebrates. Dog is feisty. Animal is large. True or false: Sally is liquid. Let us think step by step.
### Response:
Sally is insect. Insects are crustaceans. Sally is crustacean. Crustaceans are invertebrates. Sally is invertebrate. Invertebrate is animal. Sally is animal. Animal is large. Sally is large. False

### Input:
{}
### Response:
"""

TemplateCOT_4Hop_false_1step = """### Input:
Animals are cows. Fae is cow. Vertebrates are animals. Cow is kind. Feline is vertebrate. Crustaceans are large. Rex is feline. True or false: Rex is kind. Let us think step by step.
### Response:
Rex is feline. Feline is vertebrate. Vertebrates are animals. Rex is animal. Animals are cows. Cow is kind. Rex is kind. True

### Input:
Vertebrates are mammals. Dogs are vertebrates. Vertebrates are cows. Sheep is bitter. Wren is carnivore. Carnivores are dogs. Mammal is spicy. True or false: Wren is temperate. Let us think step by step.
### Response:
Wren is carnivore. Carnivores are dogs. Dogs are vertebrates. Wren is vertebrate. Vertebrates are mammals. Mammal is spicy. Wren is spicy. False

### Input:
Butterflies are animals. Dogs are mammals. Lepidopterans are temperate. Ants are lepidopterans. Rex is butterfly. Animals are ants. Mammals are snowy. True or false: Rex is temperate. Let us think step by step.
### Response:
Rex is butterfly. Butterflies are animals. Animals are ants. Rex is ant. Ants are lepidopterans. Lepidopterans are temperate. Rex is temperate. True

### Input:
Vertebrate is red. Carnivores are mammals. Polly is carnivore. Moths are discordant. Mammals are dogs. Dog is vertebrate. Spiders are moths. True or false: Polly is sweet. Let us think step by step.
### Response:
Polly is carnivore. Carnivores are mammals. Mammals are dogs. Polly is dog. Dog is vertebrate. Vertebrate is red. Polly is red. False

### Input:
Cows are snakes. Dogs are cows. Stella is cow. Max is dog. Carnivore is mammal. Snakes are carnivores. Mammals are amenable. True or false: Stella is amenable. Let us think step by step.
### Response:
Stella is cow. Cows are snakes. Snakes are carnivores. Stella is carnivore. Carnivore is mammal. Mammals are amenable. Stella is amenable. True

### Input:
Sally is insect. Insects are crustaceans. Invertebrate is animal. Feline is small. Crustaceans are invertebrates. Dog is feisty. Animal is large. True or false: Sally is liquid. Let us think step by step.
### Response:
Sally is insect. Insects are crustaceans. Crustaceans are invertebrates. Sally is invertebrate. Invertebrate is animal. Animal is large. Sally is large. False

### Input:
{}
### Response:
"""

TemplateCOT_4Hop_false_2step = """### Input:
Animals are cows. Fae is cow. Vertebrates are animals. Cow is kind. Feline is vertebrate. Crustaceans are large. Rex is feline. True or false: Rex is kind. Let us think step by step.
### Response:
Rex is feline. Feline is vertebrate. Vertebrates are animals. Animals are cows. Rex is cow. Cow is kind. Rex is kind. True

### Input:
Vertebrates are mammals. Dogs are vertebrates. Vertebrates are cows. Sheep is bitter. Wren is carnivore. Carnivores are dogs. Mammal is spicy. True or false: Wren is temperate. Let us think step by step.
### Response:
Wren is carnivore. Carnivores are dogs. Dogs are vertebrates. Vertebrates are mammals. Wren is mammal. Mammal is spicy. Wren is spicy. False

### Input:
Butterflies are animals. Dogs are mammals. Lepidopterans are temperate. Ants are lepidopterans. Rex is butterfly. Animals are ants. Mammals are snowy. True or false: Rex is temperate. Let us think step by step.
### Response:
Rex is butterfly. Butterflies are animals. Animals are ants. Ants are lepidopterans. Rex is lepidopteran. Lepidopterans are temperate. Rex is temperate. True

### Input:
Vertebrate is red. Carnivores are mammals. Polly is carnivore. Moths are discordant. Mammals are dogs. Dog is vertebrate. Spiders are moths. True or false: Polly is sweet. Let us think step by step.
### Response:
Polly is carnivore. Carnivores are mammals. Mammals are dogs. Dog is vertebrate. Polly is vertebrate. Vertebrate is red. Polly is red. False

### Input:
Cows are snakes. Dogs are cows. Stella is cow. Max is dog. Carnivore is mammal. Snakes are carnivores. Mammals are amenable. True or false: Stella is amenable. Let us think step by step.
### Response:
Stella is cow. Cows are snakes. Snakes are carnivores. Carnivore is mammal. Stella is mammal. Mammals are amenable. Stella is amenable. True

### Input:
Sally is insect. Insects are crustaceans. Invertebrate is animal. Feline is small. Crustaceans are invertebrates. Dog is feisty. Animal is large. True or false: Sally is liquid. Let us think step by step.
### Response:
Sally is insect. Insects are crustaceans. Crustaceans are invertebrates. Invertebrate is animal. Sally is animal. Animal is large. Sally is large. False

### Input:
{}
### Response:
"""
TemplateCOT_4Hop_fictional = """### Input:
Sterpuses are lempuses. Stella is zumpus. Zumpus is sterpus. Gorpus is twimpus. Lempus is rompus. Alex is rompus. Rompuses are transparent. True or false: Stella is slow. Let us think step by step.
### Response:
Stella is zumpus. Zumpus is sterpus. Stella is sterpus. Sterpuses are lempuses. Stella is lempus. Lempus is rompus. Stella is rompus. Rompuses are transparent. Stella is transparent. False

### Input:
Wren is zumpus. Shumpus is moderate. Sally is lempus. Lorpuses are shumpuses. Vumpus is lorpus. Lempus is vumpus. Gorpuses are discordant. True or false: Sally is liquid. Let us think step by step.
### Response:
Sally is lempus. Lempus is vumpus. Sally is vumpus. Vumpus is lorpus. Sally is lorpus. Lorpuses are shumpuses. Sally is shumpus. Shumpus is moderate. Sally is moderate. False

### Input:
Wumpuses are sterpuses. Jompus is fruity. Sterpus is brimpus. Impus is medium. Sterpuses are dull. Polly is wumpus. Brimpus is jompus. True or false: Polly is fruity. Let us think step by step.
### Response:
Polly is wumpus. Wumpuses are sterpuses. Polly is sterpus. Sterpus is brimpus. Polly is brimpus. Brimpus is jompus. Polly is jompus. Jompus is fruity. Polly is fruity. True

### Input:
Polly is jompus. Borpin are wumpus. Impuses are fruity. Sterpus is impus. Numpuses are sterpuses. Wumpuses are angry. Jompuses are numpuses. True or false: Polly is fruity. Let us think step by step.
### Response:
Polly is jompus. Jompuses are numpuses. Polly is numpus. Numpuses are sterpuses. Polly is sterpus. Sterpus is impus. Polly is impus. Impuses are fruity. Polly is fruity. True

### Input:
Lempuses are dumpuses. Gorpus is jelgit. Wren is lempus. Zumpuses are rompuses. Rompus is earthy. Dumpus is zumpus. Yumpuses are loud. True or false: Wren is slow. Let us think step by step.
### Response:
Wren is lempus. Lempuses are dumpuses. Wren is dumpus. Dumpus is zumpus. Wren is zumpus. Zumpuses are rompuses. Wren is rompus. Rompus is earthy. Wren is earthy. False

### Input:
Shumpuses are wumpuses. Wumpuses are muffled. Gorpus is tumpus. Max is lempus. Zumpus is dumpus. Sam is zumpus. Dumpus is shumpus. True or false: Sam is muffled. Let us think step by step.
### Response:
Sam is zumpus. Zumpus is dumpus. Sam is dumpus. Dumpus is shumpus. Sam is shumpus. Shumpuses are wumpuses. Sam is wumpus. Wumpuses are muffled. Sam is muffled. True

### Input:
{}
### Response:
"""

TemplateCOT_4Hop_fictional_1step = """### Input:
Sterpuses are lempuses. Stella is zumpus. Zumpus is sterpus. Gorpus is twimpus. Lempus is rompus. Alex is rompus. Rompuses are transparent. True or false: Stella is slow. Let us think step by step.
### Response:
Stella is zumpus. Zumpus is sterpus. Sterpuses are lempuses. Stella is lempus. Lempus is rompus. Rompuses are transparent. Stella is transparent. False

### Input:
Wren is zumpus. Shumpus is moderate. Sally is lempus. Lorpuses are shumpuses. Vumpus is lorpus. Lempus is vumpus. Gorpuses are discordant. True or false: Sally is liquid. Let us think step by step.
### Response:
Sally is lempus. Lempus is vumpus. Vumpus is lorpus. Sally is lorpus. Lorpuses are shumpuses. Shumpus is moderate. Sally is moderate. False

### Input:
Wumpuses are sterpuses. Jompus is fruity. Sterpus is brimpus. Impus is medium. Sterpuses are dull. Polly is wumpus. Brimpus is jompus. True or false: Polly is fruity. Let us think step by step.
### Response:
Polly is wumpus. Wumpuses are sterpuses. Sterpus is brimpus. Polly is brimpus. Brimpus is jompus. Jompus is fruity. Polly is fruity. True

### Input:
Polly is jompus. Borpin are wumpus. Impuses are fruity. Sterpus is impus. Numpuses are sterpuses. Wumpuses are angry. Jompuses are numpuses. True or false: Polly is fruity. Let us think step by step.
### Response:
Polly is jompus. Jompuses are numpuses. Numpuses are sterpuses. Polly is sterpus. Sterpus is impus. Impuses are fruity. Polly is fruity. True

### Input:
Lempuses are dumpuses. Gorpus is jelgit. Wren is lempus. Zumpuses are rompuses. Rompus is earthy. Dumpus is zumpus. Yumpuses are loud. True or false: Wren is slow. Let us think step by step.
### Response:
Wren is lempus. Lempuses are dumpuses. Dumpus is zumpus. Wren is zumpus. Zumpuses are rompuses. Rompus is earthy. Wren is earthy. False

### Input:
Shumpuses are wumpuses. Wumpuses are muffled. Gorpus is tumpus. Max is lempus. Zumpus is dumpus. Sam is zumpus. Dumpus is shumpus. True or false: Sam is muffled. Let us think step by step.
### Response:
Sam is zumpus. Zumpus is dumpus. Dumpus is shumpus. Sam is shumpus. Shumpuses are wumpuses. Wumpuses are muffled. Sam is muffled. True

### Input:
{}
### Response:
"""

TemplateCOT_4Hop_fictional_2step = """### Input:
Sterpuses are lempuses. Stella is zumpus. Zumpus is sterpus. Gorpus is twimpus. Lempus is rompus. Alex is rompus. Rompuses are transparent. True or false: Stella is slow. Let us think step by step.
### Response:
Stella is zumpus. Zumpus is sterpus. Sterpuses are lempuses. Lempus is rompus. Stella is rompus. Rompuses are transparent. Stella is transparent. False

### Input:
Wren is zumpus. Shumpus is moderate. Sally is lempus. Lorpuses are shumpuses. Vumpus is lorpus. Lempus is vumpus. Gorpuses are discordant. True or false: Sally is liquid. Let us think step by step.
### Response:
Sally is lempus. Lempus is vumpus. Vumpus is lorpus. Lorpuses are shumpuses. Sally is shumpus. Shumpus is moderate. Sally is moderate. False

### Input:
Wumpuses are sterpuses. Jompus is fruity. Sterpus is brimpus. Impus is medium. Sterpuses are dull. Polly is wumpus. Brimpus is jompus. True or false: Polly is fruity. Let us think step by step.
### Response:
Polly is wumpus. Wumpuses are sterpuses. Sterpus is brimpus. Brimpus is jompus. Polly is jompus. Jompus is fruity. Polly is fruity. True

### Input:
Polly is jompus. Borpin are wumpus. Impuses are fruity. Sterpus is impus. Numpuses are sterpuses. Wumpuses are angry. Jompuses are numpuses. True or false: Polly is fruity. Let us think step by step.
### Response:
Polly is jompus. Jompuses are numpuses. Numpuses are sterpuses. Sterpus is impus. Polly is impus. Impuses are fruity. Polly is fruity. True

### Input:
Lempuses are dumpuses. Gorpus is jelgit. Wren is lempus. Zumpuses are rompuses. Rompus is earthy. Dumpus is zumpus. Yumpuses are loud. True or false: Wren is slow. Let us think step by step.
### Response:
Wren is lempus. Lempuses are dumpuses. Dumpus is zumpus. Zumpuses are rompuses. Wren is rompus. Rompus is earthy. Wren is earthy. False

### Input:
Shumpuses are wumpuses. Wumpuses are muffled. Gorpus is tumpus. Max is lempus. Zumpus is dumpus. Sam is zumpus. Dumpus is shumpus. True or false: Sam is muffled. Let us think step by step.
### Response:
Sam is zumpus. Zumpus is dumpus. Dumpus is shumpus. Shumpuses are wumpuses. Sam is wumpus. Wumpuses are muffled. Sam is muffled. True

### Input:
{}
### Response:
"""
