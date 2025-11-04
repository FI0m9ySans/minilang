## 中文版描述

MiniLang 是一款完全从零开始设计的自制编程语言，专为中文用户和学习编程的初学者打造。该语言采用直观的中文关键字，让编程逻辑更加贴近自然语言表达。

**核心特性：**
- 使用中文关键字：`定义`、`函数`、`返回`、`如果`、`否则`、`当`等
- 简洁的语法结构，降低学习门槛
- 支持变量、函数、条件判断、循环等基本编程概念
- 内置输出函数 `kio` 用于显示结果
- 文件扩展名为 `.kalop`，易于识别

**语法示例：**
```
定义 数字 = 10;
函数 平方(甲) {
    返回 甲 * 甲;
}
kio("结果：", 平方(数字));
```

**技术架构：**
MiniLang 包含完整的编译器前端：词法分析器识别中文关键字和标识符，语法分析器构建抽象语法树，解释器执行代码。它使用 Python 实现，支持 Unicode 字符集，能够正确处理中文标识符和注释。

**应用场景：**
适合编程教学、算法演示、脚本编写和教育软件开发。特别适合中文环境的编程初学者，帮助他们理解编程概念而不受英语障碍影响。

## English Version Description

MiniLang is a self-designed programming language built completely from scratch, specifically crafted for Chinese users and programming beginners. It features intuitive Chinese keywords that make programming logic more aligned with natural language expression.

**Core Features:**
- Uses Chinese keywords: `定义` (define), `函数` (function), `返回` (return), `如果` (if), `否则` (else), `当` (while), etc.
- Clean syntax structure that lowers the learning curve
- Supports basic programming concepts including variables, functions, conditional statements, and loops
- Built-in output function `kio` for displaying results
- File extension `.kalop` for easy identification

**Syntax Example:**
```
定义 number = 10;
函数 square(x) {
    返回 x * x;
}
kio("Result:", square(number));
```

**Technical Architecture:**
MiniLang includes a complete compiler frontend: a lexer that recognizes Chinese keywords and identifiers, a parser that builds abstract syntax trees, and an interpreter that executes code. Implemented in Python, it supports Unicode character sets and properly handles Chinese identifiers and comments.

**Application Scenarios:**
Ideal for programming education, algorithm demonstration, script writing, and educational software development. Particularly suitable for Chinese-speaking programming beginners, helping them understand programming concepts without English language barriers. The language serves as an excellent tool for introducing fundamental computer science concepts in native language environments.
