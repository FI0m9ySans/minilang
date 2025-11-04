import re
import sys
import os

class Token:
    def __init__(self, type, value, line, column):
        self.type = type
        self.value = value
        self.line = line
        self.column = column
    
    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)}, {self.line}, {self.column})"

class Lexer:
    def __init__(self, code):
        self.code = code
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        
        # 定义词法规则 - 调整顺序，让更具体的规则在前面
        self.token_specs = [
            # 注释 - 放在最前面
            ('COMMENT_SINGLE', r'//[^\n]*'),
            ('COMMENT_MULTI', r'/\*[\s\S]*?\*/'),
            
            # 关键字 - 改为中文
            ('VAR', r'\b定义\b'),           # var -> 定义
            ('IF', r'\b如果\b'),            # if -> 如果
            ('ELSE', r'\b否则\b'),          # else -> 否则
            ('WHILE', r'\b当\b'),           # while -> 当
            ('FUNCTION', r'\b函数\b'),      # function -> 函数
            ('RETURN', r'\b返回\b'),        # return -> 返回
            ('TRUE', r'\b真\b'),           # true -> 真
            ('FALSE', r'\b假\b'),          # false -> 假
            
            # 数据类型
            ('NUMBER', r'\b\d+(\.\d*)?\b'),
            ('STRING', r'"[^"]*"'),
            # 修改标识符规则以支持中文
            ('IDENTIFIER', r'[a-zA-Z_\u4e00-\u9fa5][a-zA-Z0-9_\u4e00-\u9fa5]*'),
            
            # 运算符
            ('EQUALS', r'=='),
            ('NOT_EQUALS', r'!='),
            ('LESS_EQUAL', r'<='),
            ('GREATER_EQUAL', r'>='),
            ('ASSIGN', r'='),
            ('PLUS', r'\+'),
            ('MINUS', r'-'),
            ('MULTIPLY', r'\*'),
            ('DIVIDE', r'/'),
            ('LESS', r'<'),
            ('GREATER', r'>'),
            
            # 标点符号
            ('LPAREN', r'\('),
            ('RPAREN', r'\)'),
            ('LBRACE', r'\{'),
            ('RBRACE', r'\}'),
            ('COMMA', r','),
            ('SEMICOLON', r';'),
            
            # 空白字符 (跳过)
            ('SKIP', r'[ \t]+'),
            ('NEWLINE', r'\n'),
        ]
        
        # 编译正则表达式
        self.regex_patterns = []
        for token_type, pattern in self.token_specs:
            self.regex_patterns.append((token_type, re.compile(pattern)))
    
    def tokenize(self):
        while self.position < len(self.code):
            matched = False
            
            for token_type, regex in self.regex_patterns:
                match = regex.match(self.code, self.position)
                if match:
                    value = match.group(0)
                    
                    # 跳过空白字符
                    if token_type == 'SKIP':
                        self.position = match.end()
                        self.column += len(value)
                        matched = True
                        break
                    
                    # 跳过注释
                    elif token_type in ['COMMENT_SINGLE', 'COMMENT_MULTI']:
                        # 计算注释中的换行符数量，更新行号和列号
                        lines = value.split('\n')
                        if len(lines) > 1:
                            self.line += len(lines) - 1
                            self.column = len(lines[-1]) + 1
                        else:
                            self.column += len(value)
                        self.position = match.end()
                        matched = True
                        break
                    
                    # 处理换行
                    elif token_type == 'NEWLINE':
                        self.line += 1
                        self.column = 1
                        self.position = match.end()
                        matched = True
                        break
                    
                    # 处理其他token
                    else:
                        token = Token(token_type, value, self.line, self.column)
                        self.tokens.append(token)
                        
                        self.position = match.end()
                        self.column += len(value)
                        matched = True
                        break
            
            if not matched:
                # 无法识别的字符
                char = self.code[self.position]
                raise SyntaxError(f"无法识别的字符 '{char}' 在位置 {self.line}:{self.column}")
        
        # 添加文件结束标记
        self.tokens.append(Token('EOF', '', self.line, self.column))
        return self.tokens

# AST节点定义
class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self, statements):
        self.statements = statements
    
    def __repr__(self):
        return f"Program({self.statements})"

class VarDeclaration(ASTNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value
    
    def __repr__(self):
        return f"VarDeclaration({self.name}, {self.value})"

class Assignment(ASTNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value
    
    def __repr__(self):
        return f"Assignment({self.name}, {self.value})"

class IfStatement(ASTNode):
    def __init__(self, condition, then_branch, else_branch=None):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch
    
    def __repr__(self):
        return f"IfStatement({self.condition}, {self.then_branch}, {self.else_branch})"

class WhileStatement(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body
    
    def __repr__(self):
        return f"WhileStatement({self.condition}, {self.body})"

class FunctionDeclaration(ASTNode):
    def __init__(self, name, parameters, body):
        self.name = name
        self.parameters = parameters
        self.body = body
    
    def __repr__(self):
        return f"FunctionDeclaration({self.name}, {self.parameters}, {self.body})"

class ReturnStatement(ASTNode):
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return f"ReturnStatement({self.value})"

class BinaryOperation(ASTNode):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right
    
    def __repr__(self):
        return f"BinaryOperation({self.left}, {self.operator}, {self.right})"

class UnaryOperation(ASTNode):
    def __init__(self, operator, operand):
        self.operator = operator
        self.operand = operand
    
    def __repr__(self):
        return f"UnaryOperation({self.operator}, {self.operand})"

class Literal(ASTNode):
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return f"Literal({self.value})"

class Identifier(ASTNode):
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f"Identifier({self.name})"

class FunctionCall(ASTNode):
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments
    
    def __repr__(self):
        return f"FunctionCall({self.name}, {self.arguments})"

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.position = 0
        self.current_token = self.tokens[0] if tokens else None
    
    def advance(self):
        self.position += 1
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = None
    
    def expect(self, token_type):
        if self.current_token and self.current_token.type == token_type:
            result = self.current_token
            self.advance()
            return result
        else:
            expected = token_type
            actual = self.current_token.type if self.current_token else "EOF"
            raise SyntaxError(f"期望 {expected}, 但得到 {actual} 在位置 {self.current_token.line}:{self.current_token.column}")
    
    def parse(self):
        statements = []
        while self.current_token and self.current_token.type != 'EOF':
            statements.append(self.parse_statement())
        return Program(statements)
    
    def parse_statement(self):
        if self.current_token.type == 'VAR':
            return self.parse_var_declaration()
        elif self.current_token.type == 'IF':
            return self.parse_if_statement()
        elif self.current_token.type == 'WHILE':
            return self.parse_while_statement()
        elif self.current_token.type == 'FUNCTION':
            return self.parse_function_declaration()
        elif self.current_token.type == 'RETURN':
            return self.parse_return_statement()
        elif self.current_token.type == 'IDENTIFIER':
            # 可能是赋值或函数调用
            lookahead = self.tokens[self.position + 1] if self.position + 1 < len(self.tokens) else None
            if lookahead and lookahead.type == 'ASSIGN':
                return self.parse_assignment()
            elif lookahead and lookahead.type == 'LPAREN':
                # 函数调用作为语句
                func_call = self.parse_function_call()
                self.expect('SEMICOLON')  # 函数调用语句需要分号
                return func_call
            else:
                expr = self.parse_expression()
                # 表达式语句需要分号
                self.expect('SEMICOLON')
                return expr
        else:
            expr = self.parse_expression()
            # 表达式语句需要分号
            self.expect('SEMICOLON')
            return expr
    
    def parse_var_declaration(self):
        self.expect('VAR')
        name = self.expect('IDENTIFIER').value
        self.expect('ASSIGN')
        value = self.parse_expression()
        self.expect('SEMICOLON')
        return VarDeclaration(name, value)
    
    def parse_assignment(self):
        name = self.expect('IDENTIFIER').value
        self.expect('ASSIGN')
        value = self.parse_expression()
        self.expect('SEMICOLON')
        return Assignment(name, value)
    
    def parse_if_statement(self):
        self.expect('IF')
        self.expect('LPAREN')
        condition = self.parse_expression()
        self.expect('RPAREN')
        self.expect('LBRACE')
        then_branch = []
        while self.current_token and self.current_token.type != 'RBRACE':
            then_branch.append(self.parse_statement())
        self.expect('RBRACE')
        
        else_branch = None
        if self.current_token and self.current_token.type == 'ELSE':
            self.expect('ELSE')
            self.expect('LBRACE')
            else_branch = []
            while self.current_token and self.current_token.type != 'RBRACE':
                else_branch.append(self.parse_statement())
            self.expect('RBRACE')
        
        return IfStatement(condition, then_branch, else_branch)
    
    def parse_while_statement(self):
        self.expect('WHILE')
        self.expect('LPAREN')
        condition = self.parse_expression()
        self.expect('RPAREN')
        self.expect('LBRACE')
        body = []
        while self.current_token and self.current_token.type != 'RBRACE':
            body.append(self.parse_statement())
        self.expect('RBRACE')
        return WhileStatement(condition, body)
    
    def parse_function_declaration(self):
        self.expect('FUNCTION')
        name = self.expect('IDENTIFIER').value
        self.expect('LPAREN')
        
        parameters = []
        if self.current_token.type != 'RPAREN':
            parameters.append(self.expect('IDENTIFIER').value)
            while self.current_token.type == 'COMMA':
                self.expect('COMMA')
                parameters.append(self.expect('IDENTIFIER').value)
        
        self.expect('RPAREN')
        self.expect('LBRACE')
        
        body = []
        while self.current_token and self.current_token.type != 'RBRACE':
            body.append(self.parse_statement())
        self.expect('RBRACE')
        
        return FunctionDeclaration(name, parameters, body)
    
    def parse_return_statement(self):
        self.expect('RETURN')
        value = self.parse_expression()
        self.expect('SEMICOLON')
        return ReturnStatement(value)
    
    def parse_function_call(self):
        name = self.expect('IDENTIFIER').value
        self.expect('LPAREN')
        
        arguments = []
        if self.current_token.type != 'RPAREN':
            arguments.append(self.parse_expression())
            while self.current_token.type == 'COMMA':
                self.expect('COMMA')
                arguments.append(self.parse_expression())
        
        self.expect('RPAREN')
        return FunctionCall(name, arguments)
    
    def parse_expression(self):
        return self.parse_comparison()
    
    def parse_comparison(self):
        node = self.parse_addition()
        
        while self.current_token and self.current_token.type in ['EQUALS', 'NOT_EQUALS', 'LESS', 'GREATER', 'LESS_EQUAL', 'GREATER_EQUAL']:
            operator = self.current_token.type
            self.advance()
            node = BinaryOperation(node, operator, self.parse_addition())
        
        return node
    
    def parse_addition(self):
        node = self.parse_multiplication()
        
        while self.current_token and self.current_token.type in ['PLUS', 'MINUS']:
            operator = self.current_token.type
            self.advance()
            node = BinaryOperation(node, operator, self.parse_multiplication())
        
        return node
    
    def parse_multiplication(self):
        node = self.parse_primary()
        
        while self.current_token and self.current_token.type in ['MULTIPLY', 'DIVIDE']:
            operator = self.current_token.type
            self.advance()
            node = BinaryOperation(node, operator, self.parse_primary())
        
        return node
    
    def parse_primary(self):
        token = self.current_token
        
        if not token:
            raise SyntaxError("意外的文件结束")
        
        if token.type == 'NUMBER':
            self.advance()
            # 判断是整数还是浮点数
            if '.' in token.value:
                return Literal(float(token.value))
            else:
                return Literal(int(token.value))
        
        elif token.type == 'STRING':
            self.advance()
            # 去掉引号
            return Literal(token.value[1:-1])
        
        elif token.type == 'TRUE':
            self.advance()
            return Literal(True)
        
        elif token.type == 'FALSE':
            self.advance()
            return Literal(False)
        
        elif token.type == 'IDENTIFIER':
            # 检查是否是函数调用
            lookahead = self.tokens[self.position + 1] if self.position + 1 < len(self.tokens) else None
            if lookahead and lookahead.type == 'LPAREN':
                return self.parse_function_call()
            else:
                name = token.value
                self.advance()
                return Identifier(name)
        
        elif token.type == 'LPAREN':
            self.advance()
            node = self.parse_expression()
            self.expect('RPAREN')
            return node
        
        else:
            raise SyntaxError(f"意外的token: {token.type} 在位置 {token.line}:{token.column}")

class ReturnException(Exception):
    """用于从函数中返回值的异常"""
    def __init__(self, value):
        self.value = value

class Interpreter:
    def __init__(self):
        self.global_scope = {}
        self.functions = {}  # 存储自定义函数
        self.builtins = {}   # 存储内置函数
        self.setup_builtins()
    
    def setup_builtins(self):
        # 添加内置函数 - 保持为 kio
        self.builtins['kio'] = self.builtin_print
    
    def builtin_print(self, args):
        print(*args)
        return None
    
    def interpret(self, ast):
        if isinstance(ast, Program):
            for statement in ast.statements:
                self.execute(statement)
        else:
            self.execute(ast)
    
    def execute(self, node):
        try:
            if isinstance(node, VarDeclaration):
                value = self.evaluate(node.value)
                self.global_scope[node.name] = value
                return value
            
            elif isinstance(node, Assignment):
                value = self.evaluate(node.value)
                if node.name not in self.global_scope:
                    raise NameError(f"未定义的变量: {node.name}")
                self.global_scope[node.name] = value
                return value
            
            elif isinstance(node, IfStatement):
                condition = self.evaluate(node.condition)
                if condition:
                    for statement in node.then_branch:
                        self.execute(statement)
                elif node.else_branch:
                    for statement in node.else_branch:
                        self.execute(statement)
                return None
            
            elif isinstance(node, WhileStatement):
                while self.evaluate(node.condition):
                    for statement in node.body:
                        self.execute(statement)
                return None
            
            elif isinstance(node, FunctionDeclaration):
                self.functions[node.name] = node
                return None
            
            elif isinstance(node, ReturnStatement):
                value = self.evaluate(node.value)
                raise ReturnException(value)
            
            elif isinstance(node, FunctionCall):
                # 首先检查是否是内置函数
                if node.name in self.builtins:
                    args = [self.evaluate(arg) for arg in node.arguments]
                    return self.builtins[node.name](args)
                
                # 然后检查是否是自定义函数
                elif node.name in self.functions:
                    func = self.functions[node.name]
                    # 创建新的作用域
                    old_scope = self.global_scope.copy()
                    
                    # 设置参数
                    args = [self.evaluate(arg) for arg in node.arguments]
                    if len(args) != len(func.parameters):
                        raise TypeError(f"函数 {node.name} 期望 {len(func.parameters)} 个参数, 但得到 {len(args)}")
                    
                    for param, arg in zip(func.parameters, args):
                        self.global_scope[param] = arg
                    
                    # 执行函数体
                    result = None
                    try:
                        for statement in func.body:
                            result = self.execute(statement)
                    except ReturnException as e:
                        result = e.value
                    
                    # 恢复作用域
                    self.global_scope = old_scope
                    return result
                else:
                    raise NameError(f"未定义的函数: {node.name}")
            
            else:
                return self.evaluate(node)
        except ReturnException:
            raise  # 重新抛出ReturnException，让上层函数调用处理
        except Exception as e:
            raise e
    
    def evaluate(self, node):
        if isinstance(node, Literal):
            return node.value
        
        elif isinstance(node, Identifier):
            if node.name in self.global_scope:
                return self.global_scope[node.name]
            else:
                raise NameError(f"未定义的变量: {node.name}")
        
        elif isinstance(node, BinaryOperation):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            
            if node.operator == 'PLUS':
                return left + right
            elif node.operator == 'MINUS':
                return left - right
            elif node.operator == 'MULTIPLY':
                return left * right
            elif node.operator == 'DIVIDE':
                if right == 0:
                    raise ZeroDivisionError("除以零错误")
                return left / right
            elif node.operator == 'EQUALS':
                return left == right
            elif node.operator == 'NOT_EQUALS':
                return left != right
            elif node.operator == 'LESS':
                return left < right
            elif node.operator == 'GREATER':
                return left > right
            elif node.operator == 'LESS_EQUAL':
                return left <= right
            elif node.operator == 'GREATER_EQUAL':
                return left >= right
        
        elif isinstance(node, UnaryOperation):
            operand = self.evaluate(node.operand)
            if node.operator == 'MINUS':
                return -operand
            # 可以添加其他一元运算符
        
        elif isinstance(node, FunctionCall):
            return self.execute(node)
        
        else:
            raise TypeError(f"无法评估的节点类型: {type(node).__name__}")

class MiniLang:
    def __init__(self, debug=False):
        self.lexer = None
        self.parser = None
        self.interpreter = Interpreter()
        self.debug = debug
    
    def run(self, code):
        try:
            # 词法分析
            self.lexer = Lexer(code)
            tokens = self.lexer.tokenize()
            
            if self.debug:
                print("=== 词法分析结果 ===")
                for token in tokens:
                    print(token)
                print()
            
            # 语法分析
            self.parser = Parser(tokens)
            ast = self.parser.parse()
            
            if self.debug:
                print("=== 语法分析结果 ===")
                print(ast)
                print()
            
            # 解释执行
            self.interpreter.interpret(ast)
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
    
    def run_file(self, filename):
        """运行 .kalop 文件"""
        if not os.path.exists(filename):
            print(f"错误: 文件 '{filename}' 不存在")
            return
        
        if not filename.endswith('.kalop'):
            print(f"警告: 文件 '{filename}' 不是 .kalop 文件")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                code = f.read()
            
            print(f"=== 运行文件: {filename} ===")
            self.run(code)
            
        except Exception as e:
            print(f"读取文件时出错: {e}")

def main():
    """主函数，支持命令行参数"""
    import sys
    
    if len(sys.argv) > 1:
        # 从命令行参数获取文件名
        filename = sys.argv[1]
        debug = len(sys.argv) > 2 and sys.argv[2] == '--debug'
        
        lang = MiniLang(debug=debug)
        lang.run_file(filename)
    else:
        # 如果没有参数，显示使用方法
        print("MiniLang 解释器")
        print("使用方法:")
        print("  python minilang.py <filename.kalop> [--debug]")
        print()
        print("示例:")
        print("  python minilang.py hello.kalop")
        print("  python minilang.py math_simple.kalop --debug")
        print()
        print("或者运行内置示例:")
        
        # 运行内置示例
        lang = MiniLang(debug=False)
        sample_code = """
        kio("=== MiniLang 示例 ===");
        定义 甲 = 10;
        定义 乙 = 5;
        
        函数 加(甲, 乙) {
            返回 甲 + 乙;
        }
        
        定义 结果 = 加(甲, 乙);
        kio("计算结果:", 结果);
        
        如果 (结果 > 10) {
            kio("结果大于10");
        } 否则 {
            kio("结果小于等于10");
        }
        """
        lang.run(sample_code)

if __name__ == "__main__":
    main()
