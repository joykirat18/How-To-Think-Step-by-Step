Template_direct = """### Input:
Stella is a real number. Real numbers are natural numbers. Every natural number is black. Who is Stella?"
### Response:
black

### Input:
Sam is a cow. Every cow is a cat. Every animal is melodic. Who is Sam?
### Response:
cat

### Input:
Polly is a number. Each animal is an integer. Integers are muffled. Who is Polly?
### Response:
number

### Input:
Wren is an animal. Animals are moths. Every impus is amenable. Who is Wren?
### Response:
moths

### Input:
Fae is a cat. Each cat is a carnivore. Every carnivore is liquid. Who is Fae?
### Response:
liquid

### Input:
Max is an invertebrate. Sheeps are not animals. Animals are slow. Who is Max?
### Response:
invertebrate

### Input:
Max is a fraction. Fractions are even numbers. Every even number is spicy. Who is Max?
### Response:
spicy

### Input:
Rex is a sheep. Every sheep is a carnivore. Every borpin is not spicy. Who is Rex?
### Response:
carnivore

### Input:
Max is a mammal. Orange is a cat. Every cat is orange. Who is Max?
### Response:
mammal

### Input:
{}
### Response:
"""

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

TemplateCOT_fictional_nonShuffled = """### Input:
Gorpus is small. Rompus is gorpus. Alex is rompus. True or false: Alex is small. Let us think step by step. 
### Response:
Alex is rompus. Rompus is gorpus. Alex is gorpus. Gorpus is small. Alex is small. True

### Input:
Gorpuses are discordant. Zumpus is gorpus. Max is zumpus. True or false: Max is small. Let us think step by step. 
### Response:
Max is zumpus. Zumpus is gorpus. Max is gorpus. Gorpuses are discordant. Max is discordant. False

### Input:
Wumpuses are angry. Lempus is wumpus. Sally is lempus. True or false: Sally is floral. Let us think step by step. 
### Response:
Sally is lempus. Lempus is wumpus. Sally is wumpus. Wumpuses are angry. Sally is angry. False

### Input:
Yumpuses are loud. Gorpus is yumpus. Rex is gorpus. True or false: Rex is loud. Let us think step by step.
### Response:
Rex is gorpus. Gorpus is yumpus. Rex is yumpus. Yumpuses are loud. Rex is loud. True

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
Lorpuses are small. Lempus is lorpus. Tumpus is lempus. Rex is tumpus. True or false: Rex is muffled. Let us think step by step. 
### Response:
Rex is tumpus. Tumpus is lempus. Rex is lempus. Lempus is lorpus. Rex is lorpus. Lorpuses are small. Rex is small. False

### Input:
Yumpus is small. Grimpuses are yumpuses. Shumpus is grimpus. Max is shumpus. True or false: Max is small. Let us think step by step. 
### Response:
Max is shumpus. Shumpus is grimpus. Max is grimpus. Grimpuses are yumpuses. Max is yumpus. Yumpus is small. Max is small. True

### Input:
Brimpuses are fast. Tumpuses are brimpuses. Jompuses are tumpuses. Max is jompus. True or false: Max is fast. Let us think step by step. 
### Response:
Max is jompus. Jompuses are tumpuses. Max is tumpus. Tumpuses are brimpuses. Max is brimpus. Brimpuses are fast. Max is fast. True

### Input:
Zumpus is bright. Wumpuses are zumpuses. Yumpus is wumpus. Rex is yumpus. True or false: Rex is happy. Let us think step by step.
### Response:
Rex is yumpus. Yumpus is wumpus. Rex is wumpus. Wumpuses are zumpuses. Rex is zumpus. Zumpus is bright. Rex is bright. False

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
Snake is metallic. Cats are snakes. Vertebrate is cat. Stella is vertebrate. True or false: Stella is red. Let us think step by step.
### Response:
Stella is vertebrate. Vertebrate is cat. Stella is cat. Cats are snakes. Stella is snake. Snake is metallic. Stella is metallic. False

### Input:
Carnivores are amenable. Snake is carnivore. Cats are snakes. Sam is cat. True or false: Sam is amenable. Let us think step by step.
### Response:
Sam is cat. Cats are snakes. Sam is snake. Snake is carnivore. Sam is carnivore. Carnivores are amenable. Sam is amenable. True

### Input:
Carnivore is fast. Feline is carnivore. Cow is feline. Stella is cow. True or false: Stella is opaque. Let us think step by step.
### Response:
Stella is cow. Cow is feline. Stella is feline. Feline is carnivore. Stella is carnivore. Carnivore is fast. Stella is fast. False

### Input:
Sheep is loud. Animals are sheep. Mammal is animal. Fae is mammal. True or false: Fae is loud. Let us think step by step.
### Response:
Fae is mammal. Mammal is animal. Fae is animal. Animals are sheep. Fae is sheep. Sheep is loud. Fae is loud. True

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
Cow is kind. Animals are cows. Vertebrates are animals. Feline is vertebrate. Rex is feline. True or false: Rex is kind. Let us think step by step.
### Response:
Rex is feline. Feline is vertebrate. Rex is vertebrate. Vertebrates are animals. Rex is animal. Animals are cows. Rex is cow. Cow is kind. Rex is kind. True

### Input:
Mammal is spicy. Vertebrates are mammals. Dogs are vertebrates. Carnivores are dogs. Wren is carnivore. True or false: Wren is temperate. Let us think step by step.
### Response:
Wren is carnivore. Carnivores are dogs. Wren is dog. Dogs are vertebrates. Wren is vertebrate. Vertebrates are mammals. Wren is mammal. Mammal is spicy. Wren is spicy. False

### Input:
Lepidopterans are temperate. Ants are lepidopterans. Animals are ants. Butterflies are animals. Rex is butterfly. True or false: Rex is temperate. Let us think step by step.
### Response:
Rex is butterfly. Butterflies are animals. Rex is animal. Animals are ants. Rex is ant. Ants are lepidopterans. Rex is lepidopteran. Lepidopterans are temperate. Rex is temperate. True

### Input:
Vertebrate is red. Dog is vertebrate. Mammals are dogs. Carnivores are mammals. Polly is carnivore. True or false: Polly is sweet. Let us think step by step.
### Response:
Polly is carnivore. Carnivores are mammals. Polly is mammal. Mammals are dogs. Polly is dog. Dog is vertebrate. Polly is vertebrate. Vertebrate is red. Polly is red. False

### Input:
{}
### Response:
"""

TemplateCOT_4Hop_false_1step = """### Input:
Cow is kind. Animals are cows. Vertebrates are animals. Feline is vertebrate. Rex is feline. True or false: Rex is kind. Let us think step by step.
### Response:
Rex is feline. Feline is vertebrate. Vertebrates are animals. Rex is animal. Animals are cows. Cow is kind. Rex is kind. True

### Input:
Mammal is spicy. Vertebrates are mammals. Dogs are vertebrates. Carnivores are dogs. Wren is carnivore. True or false: Wren is temperate. Let us think step by step.
### Response:
Wren is carnivore. Carnivores are dogs. Dogs are vertebrates. Wren is vertebrate. Vertebrates are mammals. Mammal is spicy. Wren is spicy. False

### Input:
Lepidopterans are temperate. Ants are lepidopterans. Animals are ants. Butterflies are animals. Rex is butterfly. True or false: Rex is temperate. Let us think step by step.
### Response:
Rex is butterfly. Butterflies are animals. Animals are ants. Rex is ant. Ants are lepidopterans. Lepidopterans are temperate. Rex is temperate. True

### Input:
Vertebrate is red. Dog is vertebrate. Mammals are dogs. Carnivores are mammals. Polly is carnivore. True or false: Polly is sweet. Let us think step by step.
### Response:
Polly is carnivore. Carnivores are mammals. Mammals are dogs. Polly is dog. Dog is vertebrate. Vertebrate is red. Polly is red. False

### Input:
{}
### Response:
"""

TemplateCOT_4Hop_false_2step = """### Input:
Cow is kind. Animals are cows. Vertebrates are animals. Feline is vertebrate. Rex is feline. True or false: Rex is kind. Let us think step by step.
### Response:
Rex is feline. Feline is vertebrate. Vertebrates are animals. Animals are cows. Rex is cow. Cow is kind. Rex is kind. True

### Input:
Mammal is spicy. Vertebrates are mammals. Dogs are vertebrates. Carnivores are dogs. Wren is carnivore. True or false: Wren is temperate. Let us think step by step.
### Response:
Wren is carnivore. Carnivores are dogs. Dogs are vertebrates. Vertebrates are mammals. Wren is mammal. Mammal is spicy. Wren is spicy. False

### Input:
Lepidopterans are temperate. Ants are lepidopterans. Animals are ants. Butterflies are animals. Rex is butterfly. True or false: Rex is temperate. Let us think step by step.
### Response:
Rex is butterfly. Butterflies are animals. Animals are ants. Ants are lepidopterans. Rex is lepidopteran. Lepidopterans are temperate. Rex is temperate. True

### Input:
Vertebrate is red. Dog is vertebrate. Mammals are dogs. Carnivores are mammals. Polly is carnivore. True or false: Polly is sweet. Let us think step by step.
### Response:
Polly is carnivore. Carnivores are mammals. Mammals are dogs. Dog is vertebrate. Polly is vertebrate. Vertebrate is red. Polly is red. False

### Input:
{}
### Response:
"""
TemplateCOT_4Hop_fictional = """### Input:
Rompuses are transparent. Lempus is rompus. Sterpuses are lempuses. Zumpus is sterpus. Stella is zumpus. True or false: Stella is slow. Let us think step by step.
### Response:
Stella is zumpus. Zumpus is sterpus. Stella is sterpus. Sterpuses are lempuses. Stella is lempus. Lempus is rompus. Stella is rompus. Rompuses are transparent. Stella is transparent. False

### Input:
Shumpus is moderate. Lorpuses are shumpuses. Vumpus is lorpus. Lempus is vumpus. Sally is lempus. True or false: Sally is liquid. Let us think step by step.
### Response:
Sally is lempus. Lempus is vumpus. Sally is vumpus. Vumpus is lorpus. Sally is lorpus. Lorpuses are shumpuses. Sally is shumpus. Shumpus is moderate. Sally is moderate. False

### Input:
Jompus is fruity. Brimpus is jompus. Sterpus is brimpus. Wumpuses are sterpuses. Polly is wumpus. True or false: Polly is fruity. Let us think step by step.
### Response:
Polly is wumpus. Wumpuses are sterpuses. Polly is sterpus. Sterpus is brimpus. Polly is brimpus. Brimpus is jompus. Polly is jompus. Jompus is fruity. Polly is fruity. True

### Input:
Impuses are fruity. Sterpus is impus. Numpuses are sterpuses. Jompuses are numpuses. Polly is jompus. True or false: Polly is fruity. Let us think step by step.
### Response:
Polly is jompus. Jompuses are numpuses. Polly is numpus. Numpuses are sterpuses. Polly is sterpus. Sterpus is impus. Polly is impus. Impuses are fruity. Polly is fruity. True

### Input:
{}
### Response:
"""

TemplateCOT_4Hop_fictional_1step = """### Input:
Rompuses are transparent. Lempus is rompus. Sterpuses are lempuses. Zumpus is sterpus. Stella is zumpus. True or false: Stella is slow. Let us think step by step.
### Response:
Stella is zumpus. Zumpus is sterpus. Sterpuses are lempuses. Stella is lempus. Lempus is rompus. Rompuses are transparent. Stella is transparent. False

### Input:
Shumpus is moderate. Lorpuses are shumpuses. Vumpus is lorpus. Lempus is vumpus. Sally is lempus. True or false: Sally is liquid. Let us think step by step.
### Response:
Sally is lempus. Lempus is vumpus. Vumpus is lorpus. Sally is lorpus. Lorpuses are shumpuses. Shumpus is moderate. Sally is moderate. False

### Input:
Jompus is fruity. Brimpus is jompus. Sterpus is brimpus. Wumpuses are sterpuses. Polly is wumpus. True or false: Polly is fruity. Let us think step by step.
### Response:
Polly is wumpus. Wumpuses are sterpuses. Sterpus is brimpus. Polly is brimpus. Brimpus is jompus. Jompus is fruity. Polly is fruity. True

### Input:
Impuses are fruity. Sterpus is impus. Numpuses are sterpuses. Jompuses are numpuses. Polly is jompus. True or false: Polly is fruity. Let us think step by step.
### Response:
Polly is jompus. Jompuses are numpuses. Numpuses are sterpuses. Polly is sterpus. Sterpus is impus. Impuses are fruity. Polly is fruity. True

### Input:
{}
### Response:
"""

TemplateCOT_4Hop_fictional_2step = """### Input:
Rompuses are transparent. Lempus is rompus. Sterpuses are lempuses. Zumpus is sterpus. Stella is zumpus. True or false: Stella is slow. Let us think step by step.
### Response:
Stella is zumpus. Zumpus is sterpus. Sterpuses are lempuses. Lempus is rompus. Stella is rompus. Rompuses are transparent. Stella is transparent. False

### Input:
Shumpus is moderate. Lorpuses are shumpuses. Vumpus is lorpus. Lempus is vumpus. Sally is lempus. True or false: Sally is liquid. Let us think step by step.
### Response:
Sally is lempus. Lempus is vumpus. Vumpus is lorpus. Lorpuses are shumpuses. Sally is shumpus. Shumpus is moderate. Sally is moderate. False

### Input:
Jompus is fruity. Brimpus is jompus. Sterpus is brimpus. Wumpuses are sterpuses. Polly is wumpus. True or false: Polly is fruity. Let us think step by step.
### Response:
Polly is wumpus. Wumpuses are sterpuses. Sterpus is brimpus. Brimpus is jompus. Polly is jompus. Jompus is fruity. Polly is fruity. True

### Input:
Impuses are fruity. Sterpus is impus. Numpuses are sterpuses. Jompuses are numpuses. Polly is jompus. True or false: Polly is fruity. Let us think step by step.
### Response:
Polly is jompus. Jompuses are numpuses. Numpuses are sterpuses. Sterpus is impus. Polly is impus. Impuses are fruity. Polly is fruity. True

### Input:
{}
### Response:
"""

Template_simple_1hop = """### Input:
Fae is vumpus. Twimpus is medium. Vumpus is transparent. Fae is
### Response:
transparent

### Input:
Yempors are werpors. Rex is grimpus. Grimpus is temperate. Rex is
### Response:
temperate


### Input:
Lorpus is blue. Stella is zhorpant. Storpists are bitter. Stella is
### Response:
zhorpant

### Input:
Stella is wolpee. Jempuses are spicy. Wumpus is angry. Stella is
### Response:
wolpee

### Input:
Kurpor is a borpin. Rompus is temperate. Wren is rompus. Wren is 
### Response:
temperate

### Input:
Jempors are phorpuses. Alex is shampor. Lorpus is bitter. Jelgits are blue. Alex is
### Response:
shampor

### Input:
Prilpant is loud. Sam is tumpus. Kurpors are phorpuses. Sam is
### Response:
tumpus

### Input:
Daumpins are amenable. Borpin is cold. Max is brimpus. Max is
### Response:
brimpus

### Input:
Polly is shumpus. Stirpor is transparent. Zhimpor is large. Polly is
### Response:
shumpus

### Input:
{}
### Response:
"""

Template_fictional_noise_0 = """### Input:
Gorpus is twimpus. Alex is rompus. Rompus is gorpus. Gorpus is small. Rompus is mean. True or false: Alex is small. Let us think step by step.
### Response:
Stella is rompus. Rompus is gorpus. Alex is gorpus. Gorpus is small. Alex is small. True

### Input:
Gorpuses are discordant. Max is zumpus. Zumpus is shampor. Zumpus is gorpus. Gorpus is earthy. True or false: Max is small. Let us think step by step. 
### Response:
Rex is zumpus. Zumpus is gorpus. Max is gorpus. Gorpuses are discordant. Max is discordant. False

### Input:
Borpin are wumpus. Wumpuses are angry. Wumpus is jempor. Sally is lempus. Lempus is wumpus. True or false: Sally is floral. Let us think step by step. 
### Response:
Fae is lempus. Lempus is wumpus. Sally is wumpus. Wumpuses are angry. Sally is angry. False

### Input:
Gorpus is jelgit. Yumpuses are loud. Gorpus is yumpus. Yumpus is orange. Rex is gorpus. True or false: Rex is loud. Let us think step by step.
### Response:
Polly is gorpus. Gorpus is yumpus. Rex is yumpus. Yumpuses are loud. Rex is loud. True

### Input:
Lempus is tumpus. Max is lempus. Tumpus is fruity. Tumpus is bright. Lempus is dropant. True or false: Max is bright. Let us think step by step.
### Response:
Fae is lempus. Lempus is tumpus. Max is tumpus. Tumpus is bright. Max is bright. True

### Input:
Sterpuses are dull. Impus is medium. Impuses are sterpuses. Wren is impus. Sterpus is daumpin. True or false: Wren is melodic. Let us think step by step.
### Response:
Alex is impus. Impuses are sterpuses. Wren is sterpus. Sterpuses are dull. Wren is dull. False

### Input:
{}
### Response:
"""
Template_fictional_noise_1 = """### Input:
Gorpus is twimpus. Alex is rompus. Rompus is gorpus. Gorpus is small. Rompus is mean. True or false: Alex is small. Let us think step by step.
### Response:
Alex is tumpus. Rompus is gorpus. Alex is gorpus. Gorpus is small. Alex is small. True

### Input:
Gorpuses are discordant. Max is zumpus. Zumpus is shampor. Zumpus is gorpus. Gorpus is earthy. True or false: Max is small. Let us think step by step. 
### Response:
Max is rompus. Zumpus is gorpus. Max is gorpus. Gorpuses are discordant. Max is discordant. False

### Input:
Borpin are wumpus. Wumpuses are angry. Wumpus is jempor. Sally is lempus. Lempus is wumpus. True or false: Sally is floral. Let us think step by step. 
### Response:
Sally is dumpus. Lempus is wumpus. Sally is wumpus. Wumpuses are angry. Sally is angry. False

### Input:
Gorpus is jelgit. Yumpuses are loud. Gorpus is yumpus. Yumpus is orange. Rex is gorpus. True or false: Rex is loud. Let us think step by step.
### Response:
Rex is rompus. Gorpus is yumpus. Rex is yumpus. Yumpuses are loud. Rex is loud. True

### Input:
Lempus is tumpus. Max is lempus. Tumpus is fruity. Tumpus is bright. Lempus is dropant. True or false: Max is bright. Let us think step by step.
### Response:
Max is lorpus. Lempus is tumpus. Max is tumpus. Tumpus is bright. Max is bright. True

### Input:
Sterpuses are dull. Impus is medium. Impuses are sterpuses. Wren is impus. Sterpus is daumpin. True or false: Wren is melodic. Let us think step by step.
### Response:
Wren is lempus. Impuses are sterpuses. Wren is sterpus. Sterpuses are dull. Wren is dull. False

### Input:
{}
### Response:
"""

Template_fictional_noise_2 = """### Input:
Gorpus is twimpus. Alex is rompus. Rompus is gorpus. Gorpus is small. Rompus is mean. True or false: Alex is small. Let us think step by step.
### Response:
Alex is rompus. Grimpus is gorpus. Alex is gorpus. Gorpus is small. Alex is small. True

### Input:
Gorpuses are discordant. Max is zumpus. Zumpus is shampor. Zumpus is gorpus. Gorpus is earthy. True or false: Max is small. Let us think step by step. 
### Response:
Max is zumpus. Shumpus is gorpus. Max is gorpus. Gorpuses are discordant. Max is discordant. False

### Input:
Borpin are wumpus. Wumpuses are angry. Wumpus is jempor. Sally is lempus. Lempus is wumpus. True or false: Sally is floral. Let us think step by step. 
### Response:
Sally is lempus. Dumpus is wumpus. Sally is wumpus. Wumpuses are angry. Sally is angry. False

### Input:
Gorpus is jelgit. Yumpuses are loud. Gorpus is yumpus. Yumpus is orange. Rex is gorpus. True or false: Rex is loud. Let us think step by step.
### Response:
Rex is gorpus. Sterpus is yumpus. Rex is yumpus. Yumpuses are loud. Rex is loud. True

### Input:
Lempus is tumpus. Max is lempus. Tumpus is fruity. Tumpus is bright. Lempus is dropant. True or false: Max is bright. Let us think step by step.
### Response:
Max is lempus. Lorpus is tumpus. Max is tumpus. Tumpus is bright. Max is bright. True

### Input:
Sterpuses are dull. Impus is medium. Impuses are sterpuses. Wren is impus. Sterpus is daumpin. True or false: Wren is melodic. Let us think step by step.
### Response:
Wren is impus. Zumpus are sterpuses. Wren is sterpus. Sterpuses are dull. Wren is dull. False

### Input:
{}
### Response:
"""

Template_fictional_noise_3 = """### Input:
Gorpus is twimpus. Alex is rompus. Rompus is gorpus. Gorpus is small. Rompus is mean. True or false: Alex is small. Let us think step by step.
### Response:
Alex is rompus. Rompus is brimpus. Alex is gorpus. Gorpus is small. Alex is small. True

### Input:
Gorpuses are discordant. Max is zumpus. Zumpus is shampor. Zumpus is gorpus. Gorpus is earthy. True or false: Max is small. Let us think step by step. 
### Response:
Max is zumpus. Zumpus is shumpus. Max is gorpus. Gorpuses are discordant. Max is discordant. False

### Input:
Borpin are wumpus. Wumpuses are angry. Wumpus is jempor. Sally is lempus. Lempus is wumpus. True or false: Sally is floral. Let us think step by step. 
### Response:
Sally is lempus. Lempus is tumpus. Sally is wumpus. Wumpuses are angry. Sally is angry. False

### Input:
Gorpus is jelgit. Yumpuses are loud. Gorpus is yumpus. Yumpus is orange. Rex is gorpus. True or false: Rex is loud. Let us think step by step.
### Response:
Rex is gorpus. Gorpus is dumpus. Rex is yumpus. Yumpuses are loud. Rex is loud. True

### Input:
Lempus is tumpus. Max is lempus. Tumpus is fruity. Tumpus is bright. Lempus is dropant. True or false: Max is bright. Let us think step by step.
### Response:
Max is lempus. Lempus is yumpus. Max is tumpus. Tumpus is bright. Max is bright. True

### Input:
Sterpuses are dull. Impus is medium. Impuses are sterpuses. Wren is impus. Sterpus is daumpin. True or false: Wren is melodic. Let us think step by step.
### Response:
Wren is impus. Impuses are gorpus. Wren is sterpus. Sterpuses are dull. Wren is dull. False

### Input:
{}
### Response:
"""

Template_fictional_noise_4 = """### Input:
Gorpus is twimpus. Alex is rompus. Rompus is gorpus. Gorpus is small. Rompus is mean. True or false: Alex is small. Let us think step by step.
### Response:
Alex is rompus. Rompus is gorpus. Stella is gorpus. Gorpus is small. Alex is small. True

### Input:
Gorpuses are discordant. Max is zumpus. Zumpus is shampor. Zumpus is gorpus. Gorpus is earthy. True or false: Max is small. Let us think step by step. 
### Response:
Max is zumpus. Zumpus is gorpus. Sally is gorpus. Gorpuses are discordant. Max is discordant. False

### Input:
Borpin are wumpus. Wumpuses are angry. Wumpus is jempor. Sally is lempus. Lempus is wumpus. True or false: Sally is floral. Let us think step by step. 
### Response:
Sally is lempus. Lempus is wumpus. Stella is wumpus. Wumpuses are angry. Sally is angry. False

### Input:
Gorpus is jelgit. Yumpuses are loud. Gorpus is yumpus. Yumpus is orange. Rex is gorpus. True or false: Rex is loud. Let us think step by step.
### Response:
Rex is gorpus. Gorpus is yumpus. Polly is yumpus. Yumpuses are loud. Rex is loud. True

### Input:
Lempus is tumpus. Max is lempus. Tumpus is fruity. Tumpus is bright. Lempus is dropant. True or false: Max is bright. Let us think step by step.
### Response:
Max is lempus. Lempus is tumpus. Polly is tumpus. Tumpus is bright. Max is bright. True

### Input:
Sterpuses are dull. Impus is medium. Impuses are sterpuses. Wren is impus. Sterpus is daumpin. True or false: Wren is melodic. Let us think step by step.
### Response:
Wren is impus. Impuses are sterpuses. Sally is sterpus. Sterpuses are dull. Wren is dull. False

### Input:
{}
### Response:
"""

Template_fictional_noise_5 = """### Input:
Gorpus is twimpus. Alex is rompus. Rompus is gorpus. Gorpus is small. Rompus is mean. True or false: Alex is small. Let us think step by step.
### Response:
Alex is rompus. Rompus is gorpus. Alex is lempus. Gorpus is small. Alex is small. True

### Input:
Gorpuses are discordant. Max is zumpus. Zumpus is shampor. Zumpus is gorpus. Gorpus is earthy. True or false: Max is small. Let us think step by step. 
### Response:
Max is zumpus. Zumpus is gorpus. Max is impus. Gorpuses are discordant. Max is discordant. False

### Input:
Borpin are wumpus. Wumpuses are angry. Wumpus is jempor. Sally is lempus. Lempus is wumpus. True or false: Sally is floral. Let us think step by step. 
### Response:
Sally is lempus. Lempus is wumpus. Sally is dumpus. Wumpuses are angry. Sally is angry. False

### Input:
Gorpus is jelgit. Yumpuses are loud. Gorpus is yumpus. Yumpus is orange. Rex is gorpus. True or false: Rex is loud. Let us think step by step.
### Response:
Rex is gorpus. Gorpus is yumpus. Rex is dumpus. Yumpuses are loud. Rex is loud. True

### Input:
Lempus is tumpus. Max is lempus. Tumpus is fruity. Tumpus is bright. Lempus is dropant. True or false: Max is bright. Let us think step by step.
### Response:
Max is lempus. Lempus is tumpus. Max is yumpus. Tumpus is bright. Max is bright. True

### Input:
Sterpuses are dull. Impus is medium. Impuses are sterpuses. Wren is impus. Sterpus is daumpin. True or false: Wren is melodic. Let us think step by step.
### Response:
Wren is impus. Impuses are sterpuses. Wren is gorpus. Sterpuses are dull. Wren is dull. False

### Input:
{}
### Response:
"""

Template_fictional_noise_6 = """### Input:
Gorpus is twimpus. Alex is rompus. Rompus is gorpus. Gorpus is small. Rompus is mean. True or false: Alex is small. Let us think step by step.
### Response:
Alex is rompus. Rompus is gorpus. Alex is gorpus. Sterpus is small. Alex is small. True

### Input:
Gorpuses are discordant. Max is zumpus. Zumpus is shampor. Zumpus is gorpus. Gorpus is earthy. True or false: Max is small. Let us think step by step. 
### Response:
Max is zumpus. Zumpus is gorpus. Max is gorpus. Lempus are discordant. Max is discordant. False

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
Max is lempus. Lempus is tumpus. Max is tumpus. Wumpus is bright. Max is bright. True

### Input:
Sterpuses are dull. Impus is medium. Impuses are sterpuses. Wren is impus. Sterpus is daumpin. True or false: Wren is melodic. Let us think step by step.
### Response:
Wren is impus. Impuses are sterpuses. Wren is sterpus. Wumpuses are dull. Wren is dull. False

### Input:
{}
### Response:
"""

Template_fictional_noise_7 = """### Input:
Gorpus is twimpus. Alex is rompus. Rompus is gorpus. Gorpus is small. Rompus is mean. True or false: Alex is small. Let us think step by step.
### Response:
Alex is rompus. Rompus is gorpus. Alex is gorpus. Gorpus is sunny. Alex is small. True

### Input:
Gorpuses are discordant. Max is zumpus. Zumpus is shampor. Zumpus is gorpus. Gorpus is earthy. True or false: Max is small. Let us think step by step. 
### Response:
Max is zumpus. Zumpus is gorpus. Max is gorpus. Gorpuses are shy. Max is discordant. False

### Input:
Borpin are wumpus. Wumpuses are angry. Wumpus is jempor. Sally is lempus. Lempus is wumpus. True or false: Sally is floral. Let us think step by step. 
### Response:
Sally is lempus. Lempus is wumpus. Sally is wumpus. Wumpuses are slow. Sally is angry. False

### Input:
Gorpus is jelgit. Yumpuses are loud. Gorpus is yumpus. Yumpus is orange. Rex is gorpus. True or false: Rex is loud. Let us think step by step.
### Response:
Rex is gorpus. Gorpus is yumpus. Rex is yumpus. Yumpuses are red. Rex is loud. True

### Input:
Lempus is tumpus. Max is lempus. Tumpus is fruity. Tumpus is bright. Lempus is dropant. True or false: Max is bright. Let us think step by step.
### Response:
Max is lempus. Lempus is tumpus. Max is tumpus. Tumpus is floral. Max is bright. True

### Input:
Sterpuses are dull. Impus is medium. Impuses are sterpuses. Wren is impus. Sterpus is daumpin. True or false: Wren is melodic. Let us think step by step.
### Response:
Wren is impus. Impuses are sterpuses. Wren is sterpus. Sterpuses are orange. Wren is dull. False

### Input:
{}
### Response:
"""

Template_fictional_noise_8 = """### Input:
Gorpus is twimpus. Alex is rompus. Rompus is gorpus. Gorpus is small. Rompus is mean. True or false: Alex is small. Let us think step by step.
### Response:
Alex is rompus. Rompus is gorpus. Alex is gorpus. Gorpus is small. Max is small. True

### Input:
Gorpuses are discordant. Max is zumpus. Zumpus is shampor. Zumpus is gorpus. Gorpus is earthy. True or false: Max is small. Let us think step by step. 
### Response:
Max is zumpus. Zumpus is gorpus. Max is gorpus. Gorpuses are discordant. Sam is discordant. False

### Input:
Borpin are wumpus. Wumpuses are angry. Wumpus is jempor. Sally is lempus. Lempus is wumpus. True or false: Sally is floral. Let us think step by step. 
### Response:
Sally is lempus. Lempus is wumpus. Sally is wumpus. Wumpuses are angry. Sam is angry. False

### Input:
Gorpus is jelgit. Yumpuses are loud. Gorpus is yumpus. Yumpus is orange. Rex is gorpus. True or false: Rex is loud. Let us think step by step.
### Response:
Rex is gorpus. Gorpus is yumpus. Rex is yumpus. Yumpuses are loud. Sam is loud. True

### Input:
Lempus is tumpus. Max is lempus. Tumpus is fruity. Tumpus is bright. Lempus is dropant. True or false: Max is bright. Let us think step by step.
### Response:
Max is lempus. Lempus is tumpus. Max is tumpus. Tumpus is bright. Fae is bright. True

### Input:
Sterpuses are dull. Impus is medium. Impuses are sterpuses. Wren is impus. Sterpus is daumpin. True or false: Wren is melodic. Let us think step by step.
### Response:
Wren is impus. Impuses are sterpuses. Wren is sterpus. Sterpuses are dull. Rex is dull. False

### Input:
{}
### Response:
"""

Template_fictional_noise_9 = """### Input:
Gorpus is twimpus. Alex is rompus. Rompus is gorpus. Gorpus is small. Rompus is mean. True or false: Alex is small. Let us think step by step.
### Response:
Alex is rompus. Rompus is gorpus. Alex is gorpus. Gorpus is small. Alex is sweet. True

### Input:
Gorpuses are discordant. Max is zumpus. Zumpus is shampor. Zumpus is gorpus. Gorpus is earthy. True or false: Max is small. Let us think step by step. 
### Response:
Max is zumpus. Zumpus is gorpus. Max is gorpus. Gorpuses are discordant. Max is fast. False

### Input:
Borpin are wumpus. Wumpuses are angry. Wumpus is jempor. Sally is lempus. Lempus is wumpus. True or false: Sally is floral. Let us think step by step. 
### Response:
Sally is lempus. Lempus is wumpus. Sally is wumpus. Wumpuses are angry. Sally is transparent. False

### Input:
Gorpus is jelgit. Yumpuses are loud. Gorpus is yumpus. Yumpus is orange. Rex is gorpus. True or false: Rex is loud. Let us think step by step.
### Response:
Rex is gorpus. Gorpus is yumpus. Rex is yumpus. Yumpuses are loud. Rex is sunny. True

### Input:
Lempus is tumpus. Max is lempus. Tumpus is fruity. Tumpus is bright. Lempus is dropant. True or false: Max is bright. Let us think step by step.
### Response:
Max is lempus. Lempus is tumpus. Max is tumpus. Tumpus is bright. Max is brown. True

### Input:
Sterpuses are dull. Impus is medium. Impuses are sterpuses. Wren is impus. Sterpus is daumpin. True or false: Wren is melodic. Let us think step by step.
### Response:
Wren is impus. Impuses are sterpuses. Wren is sterpus. Sterpuses are dull. Wren is moderate. False

### Input:
{}
### Response:
"""

Template_false_noise_0 = """### Input:
Cats are cows. Spider are vertebrate. Rex is cat. Cows are large. Cows are large. True or false: Rex is large. Let us think step by step.
### Response:
Sally is cat. Cats are cows. Rex is cow. Cows are large. Rex is large. True

### Input:
Vertebrates are cows. Sheep is bitter. Cows are sheep. Fae is cow. Crustaceans are large. True or false: Fae is red. Let us think step by step.
### Response:
Polly is cow. Cows are sheep. Fae is sheep. Sheep is bitter. Fae is bitter. False

### Input:
Dogs are mammals. Mammals are snowy. Lepidopterans are crustaceans. Alex is dog. Mammals are arthropods. True or false: Alex is snowy. Let us think step by step.
### Response:
Sam is dog. Dogs are mammals. Alex is mammal. Mammals are snowy. Alex is snowy. True

### Input:
Moths are discordant. Spiders are moths. Sam is mammal. Mammals are snakes. Snakes are spicy. True or false: Sam is spicy. Let us think step by step.
### Response:
Stella is mammal. Mammals are snakes. Sam is snake. Snakes are spicy. Sam is spicy. True

### Input:
Dogs are cows. Max is dog. Butterflies are spiders. Cows are dull. Spiders are floral. True or false: Max is luminous. Let us think step by step.
### Response:
Alex is dog. Dogs are cows. Max is cow. Cows are dull. Max is dull. False

### Input:
Dog is feisty. Feline is small. Sheep are melodic. Sam is mammal. Mammals are dogs. True or false: Sam is feisty. Let us think step by step.
### Response:
Wren is mammal. Mammals are dogs. Sam is dog. Dog is feisty. Sam is feisty. True

### Input:
{}
### Response:
"""
Template_false_noise_1 = """### Input:
Cats are cows. Spider are vertebrate. Rex is cat. Cows are large. Cows are large. True or false: Rex is large. Let us think step by step.
### Response:
Rex is Sheep. Cats are cows. Rex is cow. Cows are large. Rex is large. True

### Input:
Vertebrates are cows. Sheep is bitter. Cows are sheep. Fae is cow. Crustaceans are large. True or false: Fae is red. Let us think step by step.
### Response:
Fae is butterfly. Cows are sheep. Fae is sheep. Sheep is bitter. Fae is bitter. False

### Input:
Dogs are mammals. Mammals are snowy. Lepidopterans are crustaceans. Alex is dog. Mammals are arthropods. True or false: Alex is snowy. Let us think step by step.
### Response:
Alex is sheep. Dogs are mammals. Alex is mammal. Mammals are snowy. Alex is snowy. True

### Input:
Moths are discordant. Spiders are moths. Sam is mammal. Mammals are snakes. Snakes are spicy. True or false: Sam is spicy. Let us think step by step.
### Response:
Sam is crustacean. Mammals are snakes. Sam is snake. Snakes are spicy. Sam is spicy. True

### Input:
Dogs are cows. Max is dog. Butterflies are spiders. Cows are dull. Spiders are floral. True or false: Max is luminous. Let us think step by step.
### Response:
Max is snake. Dogs are cows. Max is cow. Cows are dull. Max is dull. False

### Input:
Dog is feisty. Feline is small. Sheep are melodic. Sam is mammal. Mammals are dogs. True or false: Sam is feisty. Let us think step by step.
### Response:
Sam is moth. Mammals are dogs. Sam is dog. Dog is feisty. Sam is feisty. True

### Input:
{}
### Response:
"""
Template_false_noise_2 = """### Input:
Cats are cows. Spider are vertebrate. Rex is cat. Cows are large. Cows are large. True or false: Rex is large. Let us think step by step.
### Response:
Rex is cat. Snakes are cows. Rex is cow. Cows are large. Rex is large. True

### Input:
Vertebrates are cows. Sheep is bitter. Cows are sheep. Fae is cow. Crustaceans are large. True or false: Fae is red. Let us think step by step.
### Response:
Fae is cow. Insect are sheep. Fae is sheep. Sheep is bitter. Fae is bitter. False

### Input:
Dogs are mammals. Mammals are snowy. Lepidopterans are crustaceans. Alex is dog. Mammals are arthropods. True or false: Alex is snowy. Let us think step by step.
### Response:
Alex is dog. Ant are mammals. Alex is mammal. Mammals are snowy. Alex is snowy. True

### Input:
Moths are discordant. Spiders are moths. Sam is mammal. Mammals are snakes. Snakes are spicy. True or false: Sam is spicy. Let us think step by step.
### Response:
Sam is mammal. Spiders are snakes. Sam is snake. Snakes are spicy. Sam is spicy. True

### Input:
Dogs are cows. Max is dog. Butterflies are spiders. Cows are dull. Spiders are floral. True or false: Max is luminous. Let us think step by step.
### Response:
Max is dog. Feline are cows. Max is cow. Cows are dull. Max is dull. False

### Input:
Dog is feisty. Feline is small. Sheep are melodic. Sam is mammal. Mammals are dogs. True or false: Sam is feisty. Let us think step by step.
### Response:
Sam is mammal. Cats are dogs. Sam is dog. Dog is feisty. Sam is feisty. True

### Input:
{}
### Response:
"""
Template_false_noise_3 = """### Input:
Cats are cows. Spider are vertebrate. Rex is cat. Cows are large. Cows are large. True or false: Rex is large. Let us think step by step.
### Response:
Rex is cat. Cats are sheeps. Rex is cow. Cows are large. Rex is large. True

### Input:
Vertebrates are cows. Sheep is bitter. Cows are sheep. Fae is cow. Crustaceans are large. True or false: Fae is red. Let us think step by step.
### Response:
Fae is cow. Cows are moth. Fae is sheep. Sheep is bitter. Fae is bitter. False

### Input:
Dogs are mammals. Mammals are snowy. Lepidopterans are crustaceans. Alex is dog. Mammals are arthropods. True or false: Alex is snowy. Let us think step by step.
### Response:
Alex is dog. Dogs are cows. Alex is mammal. Mammals are snowy. Alex is snowy. True

### Input:
Moths are discordant. Spiders are moths. Sam is mammal. Mammals are snakes. Snakes are spicy. True or false: Sam is spicy. Let us think step by step.
### Response:
Sam is mammal. Mammals are ants. Sam is snake. Snakes are spicy. Sam is spicy. True

### Input:
Dogs are cows. Max is dog. Butterflies are spiders. Cows are dull. Spiders are floral. True or false: Max is luminous. Let us think step by step.
### Response:
Max is dog. Dogs are snake. Max is cow. Cows are dull. Max is dull. False

### Input:
Dog is feisty. Feline is small. Sheep are melodic. Sam is mammal. Mammals are dogs. True or false: Sam is feisty. Let us think step by step.
### Response:
Sam is mammal. Mammals are insects. Sam is dog. Dog is feisty. Sam is feisty. True

### Input:
{}
### Response:
"""
Template_false_noise_4 = """### Input:
Cats are cows. Spider are vertebrate. Rex is cat. Cows are large. Cows are large. True or false: Rex is large. Let us think step by step.
### Response:
Rex is cat. Cats are cows. Max is cow. Cows are large. Rex is large. True

### Input:
Vertebrates are cows. Sheep is bitter. Cows are sheep. Fae is cow. Crustaceans are large. True or false: Fae is red. Let us think step by step.
### Response:
Fae is cow. Cows are sheep. Alex is sheep. Sheep is bitter. Fae is bitter. False

### Input:
Dogs are mammals. Mammals are snowy. Lepidopterans are crustaceans. Alex is dog. Mammals are arthropods. True or false: Alex is snowy. Let us think step by step.
### Response:
Alex is dog. Dogs are mammals. Sam is mammal. Mammals are snowy. Alex is snowy. True

### Input:
Moths are discordant. Spiders are moths. Sam is mammal. Mammals are snakes. Snakes are spicy. True or false: Sam is spicy. Let us think step by step.
### Response:
Sam is mammal. Mammals are snakes. Sally is snake. Snakes are spicy. Sam is spicy. True

### Input:
Dogs are cows. Max is dog. Butterflies are spiders. Cows are dull. Spiders are floral. True or false: Max is luminous. Let us think step by step.
### Response:
Max is dog. Dogs are cows. Stella is cow. Cows are dull. Max is dull. False

### Input:
Dog is feisty. Feline is small. Sheep are melodic. Sam is mammal. Mammals are dogs. True or false: Sam is feisty. Let us think step by step.
### Response:
Sam is mammal. Mammals are dogs. Polly is dog. Dog is feisty. Sam is feisty. True

### Input:
{}
### Response:
"""
Template_false_noise_5 = """### Input:
Cats are cows. Spider are vertebrate. Rex is cat. Cows are large. Cows are large. True or false: Rex is large. Let us think step by step.
### Response:
Rex is cat. Cats are cows. Rex is invertebrate. Cows are large. Rex is large. True

### Input:
Vertebrates are cows. Sheep is bitter. Cows are sheep. Fae is cow. Crustaceans are large. True or false: Fae is red. Let us think step by step.
### Response:
Fae is cow. Cows are sheep. Fae is animal. Sheep is bitter. Fae is bitter. False

### Input:
Dogs are mammals. Mammals are snowy. Lepidopterans are crustaceans. Alex is dog. Mammals are arthropods. True or false: Alex is snowy. Let us think step by step.
### Response:
Alex is dog. Dogs are mammals. Alex is carnivore. Mammals are snowy. Alex is snowy. True

### Input:
Moths are discordant. Spiders are moths. Sam is mammal. Mammals are snakes. Snakes are spicy. True or false: Sam is spicy. Let us think step by step.
### Response:
Sam is mammal. Mammals are snakes. Sam is cow. Snakes are spicy. Sam is spicy. True

### Input:
Dogs are cows. Max is dog. Butterflies are spiders. Cows are dull. Spiders are floral. True or false: Max is luminous. Let us think step by step.
### Response:
Max is dog. Dogs are cows. Max is ant. Cows are dull. Max is dull. False

### Input:
Dog is feisty. Feline is small. Sheep are melodic. Sam is mammal. Mammals are dogs. True or false: Sam is feisty. Let us think step by step.
### Response:
Sam is mammal. Mammals are dogs. Sam is crustacean. Dog is feisty. Sam is feisty. True

### Input:
{}
### Response:
"""
Template_false_noise_6 = """### Input:
Cats are cows. Spider are vertebrate. Rex is cat. Cows are large. Cows are large. True or false: Rex is large. Let us think step by step.
### Response:
Rex is cat. Cats are cows. Rex is cow. Dogs are large. Rex is large. True

### Input:
Vertebrates are cows. Sheep is bitter. Cows are sheep. Fae is cow. Crustaceans are large. True or false: Fae is red. Let us think step by step.
### Response:
Fae is cow. Cows are sheep. Fae is sheep. Feline is bitter. Fae is bitter. False

### Input:
Dogs are mammals. Mammals are snowy. Lepidopterans are crustaceans. Alex is dog. Mammals are arthropods. True or false: Alex is snowy. Let us think step by step.
### Response:
Alex is dog. Dogs are mammals. Alex is mammal. Spider are snowy. Alex is snowy. True

### Input:
Moths are discordant. Spiders are moths. Sam is mammal. Mammals are snakes. Snakes are spicy. True or false: Sam is spicy. Let us think step by step.
### Response:
Sam is mammal. Mammals are snakes. Sam is snake. Butterfly are spicy. Sam is spicy. True

### Input:
Dogs are cows. Max is dog. Butterflies are spiders. Cows are dull. Spiders are floral. True or false: Max is luminous. Let us think step by step.
### Response:
Max is dog. Dogs are cows. Max is cow. Moths are dull. Max is dull. False

### Input:
Dog is feisty. Feline is small. Sheep are melodic. Sam is mammal. Mammals are dogs. True or false: Sam is feisty. Let us think step by step.
### Response:
Sam is mammal. Mammals are dogs. Sam is dog. Carnivore is feisty. Sam is feisty. True

### Input:
{}
### Response:
"""
Template_false_noise_7 = """### Input:
Cats are cows. Spider are vertebrate. Rex is cat. Cows are large. Cows are large. True or false: Rex is large. Let us think step by step.
### Response:
Rex is cat. Cats are cows. Rex is cow. Cows are kind. Rex is large. True

### Input:
Vertebrates are cows. Sheep is bitter. Cows are sheep. Fae is cow. Crustaceans are large. True or false: Fae is red. Let us think step by step.
### Response:
Fae is cow. Cows are sheep. Fae is sheep. Sheep is snowy. Fae is bitter. False

### Input:
Dogs are mammals. Mammals are snowy. Lepidopterans are crustaceans. Alex is dog. Mammals are arthropods. True or false: Alex is snowy. Let us think step by step.
### Response:
Alex is dog. Dogs are mammals. Alex is mammal. Mammals are sad. Alex is snowy. True

### Input:
Moths are discordant. Spiders are moths. Sam is mammal. Mammals are snakes. Snakes are spicy. True or false: Sam is spicy. Let us think step by step.
### Response:
Sam is mammal. Mammals are snakes. Sam is snake. Snakes are cold. Sam is spicy. True

### Input:
Dogs are cows. Max is dog. Butterflies are spiders. Cows are dull. Spiders are floral. True or false: Max is luminous. Let us think step by step.
### Response:
Max is dog. Dogs are cows. Max is cow. Cows are shy. Max is dull. False

### Input:
Dog is feisty. Feline is small. Sheep are melodic. Sam is mammal. Mammals are dogs. True or false: Sam is feisty. Let us think step by step.
### Response:
Sam is mammal. Mammals are dogs. Sam is dog. Dog is bitter. Sam is feisty. True

### Input:
{}
### Response:
"""
Template_false_noise_8 = """### Input:
Cats are cows. Spider are vertebrate. Rex is cat. Cows are large. Cows are large. True or false: Rex is large. Let us think step by step.
### Response:
Rex is cat. Cats are cows. Rex is cow. Cows are large. Alex is large. True

### Input:
Vertebrates are cows. Sheep is bitter. Cows are sheep. Fae is cow. Crustaceans are large. True or false: Fae is red. Let us think step by step.
### Response:
Fae is cow. Cows are sheep. Fae is sheep. Sheep is bitter. Sally is bitter. False

### Input:
Dogs are mammals. Mammals are snowy. Lepidopterans are crustaceans. Alex is dog. Mammals are arthropods. True or false: Alex is snowy. Let us think step by step.
### Response:
Alex is dog. Dogs are mammals. Alex is mammal. Mammals are snowy. Sally is snowy. True

### Input:
Moths are discordant. Spiders are moths. Sam is mammal. Mammals are snakes. Snakes are spicy. True or false: Sam is spicy. Let us think step by step.
### Response:
Sam is mammal. Mammals are snakes. Sam is snake. Snakes are spicy. Polly is spicy. True

### Input:
Dogs are cows. Max is dog. Butterflies are spiders. Cows are dull. Spiders are floral. True or false: Max is luminous. Let us think step by step.
### Response:
Max is dog. Dogs are cows. Max is cow. Cows are dull. Fae is dull. False

### Input:
Dog is feisty. Feline is small. Sheep are melodic. Sam is mammal. Mammals are dogs. True or false: Sam is feisty. Let us think step by step.
### Response:
Sam is mammal. Mammals are dogs. Sam is dog. Dog is feisty. Polly is feisty. True

### Input:
{}
### Response:
"""
Template_false_noise_9 = """### Input:
Cats are cows. Spider are vertebrate. Rex is cat. Cows are large. Cows are large. True or false: Rex is large. Let us think step by step.
### Response:
Rex is cat. Cats are cows. Rex is cow. Cows are large. Rex is rainy. True

### Input:
Vertebrates are cows. Sheep is bitter. Cows are sheep. Fae is cow. Crustaceans are large. True or false: Fae is red. Let us think step by step.
### Response:
Fae is cow. Cows are sheep. Fae is sheep. Sheep is bitter. Fae is red. False

### Input:
Dogs are mammals. Mammals are snowy. Lepidopterans are crustaceans. Alex is dog. Mammals are arthropods. True or false: Alex is snowy. Let us think step by step.
### Response:
Alex is dog. Dogs are mammals. Alex is mammal. Mammals are snowy. Alex is earthy. True

### Input:
Moths are discordant. Spiders are moths. Sam is mammal. Mammals are snakes. Snakes are spicy. True or false: Sam is spicy. Let us think step by step.
### Response:
Sam is mammal. Mammals are snakes. Sam is snake. Snakes are spicy. Sam is dull. True

### Input:
Dogs are cows. Max is dog. Butterflies are spiders. Cows are dull. Spiders are floral. True or false: Max is luminous. Let us think step by step.
### Response:
Max is dog. Dogs are cows. Max is cow. Cows are dull. Max is happy. False

### Input:
Dog is feisty. Feline is small. Sheep are melodic. Sam is mammal. Mammals are dogs. True or false: Sam is feisty. Let us think step by step.
### Response:
Sam is mammal. Mammals are dogs. Sam is dog. Dog is feisty. Sam is fast. True

### Input:
{}
### Response:
"""
