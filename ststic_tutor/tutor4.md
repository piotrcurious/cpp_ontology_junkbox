To optimize non-destructible classes and methods in embedded systems using GCC, you should focus on making sure that the compiler knows the class or object will never be destroyed and its code or data can be shared across the program to minimize both Flash (program) and RAM usage.

### Key Techniques to Optimize Non-Destructible Classes and Methods

#### 1. **Use `constexpr` for Compile-Time Constants**

When a class or its methods can be fully evaluated at compile-time, use `constexpr`. This tells the compiler that the class or its members should be evaluated at compile time and stored in Flash memory, reducing runtime overhead and RAM usage.

**Example: Immutable Class using `constexpr`**

```cpp
class ImmutableConfig {
public:
    constexpr ImmutableConfig(int baud, int timeout)
        : baudRate(baud), timeoutMs(timeout) {}

    constexpr int getBaudRate() const { return baudRate; }
    constexpr int getTimeout() const { return timeoutMs; }

private:
    const int baudRate;
    const int timeoutMs;
};

// Object will be placed in Flash memory
constexpr ImmutableConfig defaultConfig(9600, 1000);
```

This object will be placed in the `.rodata` section (read-only data section) in Flash memory, allowing it to be accessed without consuming RAM.

#### 2. **Use `static` Classes and Methods**

Declaring methods or classes as `static` ensures they are shared and do not require an instance. This is particularly useful for utility functions that do not require any class state.

**Example: Static Utility Class**

```cpp
class MathUtils {
public:
    static int multiply(int a, int b) {
        return a * b;
    }
};
```

This class does not have any state and its `multiply` method can be optimized by the compiler. Since it is `static`, it can be shared across translation units without requiring an object instance.

#### 3. **Place Non-Destructible Objects in Flash Memory with `__attribute__((section))`**

If you have a non-destructible object that must be stored in Flash, use GCC's `__attribute__((section))` to explicitly place it in a designated section.

**Example: Placing an Object in Flash Memory**

```cpp
const char configData[] __attribute__((section(".config_data"))) = "BaudRate: 9600\nTimeout: 1000";
```

By placing the `configData` array in a custom section (`.config_data`), you can control exactly where it resides in Flash memory, keeping it out of RAM.

#### 4. **Use `__attribute__((const))` or `__attribute__((pure))` for Functions**

These attributes help the compiler understand that functions do not have side effects, which allows for greater optimization.

- `__attribute__((const))`: Indicates that the function has no side effects and does not access any global or static variables.
- `__attribute__((pure))`: Indicates that the function has no side effects but might read global or static variables.

**Example: Const Function**

```cpp
int getConstantValue() __attribute__((const));
int getConstantValue() {
    return 42;
}
```

This tells the compiler that the function always returns the same output for the same input, and it does not modify any state, allowing it to optimize it aggressively.

#### 5. **Use the `-ffunction-sections` and `-fdata-sections` Flags**

By using `-ffunction-sections` and `-fdata-sections`, the compiler places each function and data item in its own section. When linked with the `-Wl,--gc-sections` linker flag, this allows the linker to remove any unused functions or data.

```sh
g++ -ffunction-sections -fdata-sections -Wl,--gc-sections main.cpp -o main
```

This setup helps reduce the binary size by removing any unused code or data, which is critical for resource-constrained embedded systems.

#### 6. **Use Linker Scripts to Organize Memory Sections**

You can use custom linker scripts to precisely control where different parts of your program are placed in memory. For example, you might want to put all read-only data and non-destructible objects into a specific section of Flash memory.

**Example: Linker Script (`.ld` file)**

```ld
SECTIONS {
    .text : {
        *(.text)         /* All code */
    } > FLASH

    .rodata : {
        *(.rodata)       /* All read-only data */
    } > FLASH

    .config_data : {
        KEEP(*(.config_data))  /* Custom data section */
    } > FLASH

    .bss : {
        *(.bss)          /* Uninitialized data */
    } > RAM

    .data : {
        *(.data)         /* Initialized data */
    } > RAM
}
```

This linker script specifies where each section should be placed. Using `KEEP` ensures that the `.config_data` section is retained and not removed during optimization.

#### 7. **Use `static` Storage Duration for Immutable Global Objects**

For objects that are global but should never be destroyed or reinitialized, declare them with `static` storage duration. This ensures the object is initialized once and stored in a specific memory location.

**Example: Static Global Object**

```cpp
static const MyClass myGlobalObject(42); // Only initialized once, stored in Flash
```

This object is allocated once, stored in Flash, and can be reused across different parts of the code.

### Embedded System Use Cases for Optimizing Non-Destructible Classes

#### Use Case 1: Hardware Configuration Objects

In embedded systems, hardware configurations are often set once and remain constant throughout the application’s runtime. Use `constexpr` or `const` objects for such configurations.

```cpp
constexpr struct UARTConfig {
    int baudRate;
    int dataBits;
    int stopBits;
} uartDefaultConfig = {9600, 8, 1}; // Stored in Flash
```

#### Use Case 2: Lookup Tables for Fast Computation

Lookup tables are common in embedded systems for applications like signal processing or data compression. These tables are typically read-only and should be stored in Flash.

```cpp
const uint8_t sinLookupTable[256] __attribute__((section(".rodata"))) = {
    // Precomputed sine values...
};
```

### Conclusion

To optimize non-destructible classes and methods in embedded systems with GCC, use `constexpr`, `static`, and attributes like `__attribute__((const))` to help the compiler understand which objects and methods are immutable or have no side effects. Use compiler flags like `-ffunction-sections`, `-fdata-sections`, and linker scripts to control the placement of data and code in memory, allowing for greater optimization of RAM and Flash usage. These techniques help achieve efficient memory usage and performance in resource-constrained environments typical of embedded systems.
