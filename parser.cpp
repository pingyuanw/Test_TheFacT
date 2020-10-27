
#include "frontend/parser.h"
#include <iostream>

namespace cs160::frontend {

void Parser::matchToken(const TokenType& tok) {
  if (nextToken() && nextToken().value().type() == tok) {
    head++;
  } else {
    throw InvalidASTError();
  }
}

std::optional<Token> Parser::nextToken(int peek) {
  if (head == (int)tokens.size() - 1) {
    return std::nullopt;
  } else {
    return tokens[head + peek];
  }
}
// E ::= id | (E) | E*E | E+E

// E ::= E + F | F
// F ::= F * G | G
// G ::= (E) | id

// E  ::= FE'
// E' ::= + FE' | ε
// F  ::= GF'
// F' ::= * GF' | ε
// G  ::= (E) | id

// E  ::= FE'
ArithmeticExprP Parser::E() {
  auto f = F();
  auto e_prime = E_prime();
  if (e_prime.first && e_prime.first.value() == PlusToken) {
    return std::make_unique<const AddExpr>(std::move(f), std::move(e_prime.second));
  }
  else {
    return f;
  }
}

// E' ::= + FE' | ε
std::pair<std::optional<Token>, ArithmeticExprP> Parser::E_prime() {

  if (nextToken() && nextToken().value() == PlusToken) {
    matchToken(PlusToken.type());
    auto f = F();
    auto e_prime = E_prime();
    if (e_prime.first && e_prime.first.value() == PlusToken) {
      // + F +x
      auto term = std::make_unique<const AddExpr>(std::move(f), std::move(e_prime.second));
      return std::make_pair(PlusToken, std::move(term));
    } else {
      // + F ε
      return std::make_pair(PlusToken, std::move(f));
    }
  } else {
    // ε
    return std::make_pair(std::nullopt, nullptr);
  }
}

// F  ::= GF'
ArithmeticExprP Parser::F() {
  auto g = G();
  auto f_prime = F_prime();
  if (f_prime.first && f_prime.first.value() == TimesToken) {
    return std::make_unique<const MultiplyExpr>(std::move(g), std::move(f_prime.second));
  }
  else {
    return g;
  }
}

// F' ::= * GF' | ε
std::pair<std::optional<Token>, ArithmeticExprP> Parser::F_prime() {
  if (nextToken() && nextToken().value() == TimesToken) {
    matchToken(TimesToken.type());
    auto g = G();
    auto f_prime = F_prime();
    if (f_prime.first && f_prime.first.value() == TimesToken) {
      // * G *x
      auto factor = std::make_unique<const MultiplyExpr>(std::move(g), std::move(f_prime.second));
      return std::make_pair(TimesToken, std::move(factor));
    } else {
      // * G ε
      return std::make_pair(TimesToken, std::move(g));
    }
  } else {
    // ε
    return std::make_pair(std::nullopt, nullptr);
  }
}

// G  ::= (E) | id
ArithmeticExprP Parser::G() {
  if (nextToken() && nextToken().value().type() == TokenType::LParen) {
    matchToken(TokenType::LParen);
    auto e = E();
    matchToken(TokenType::RParen);
    return e;
  } else if (nextToken() && nextToken().value().type() == TokenType::Id) {
    return ID();
  }
  throw InvalidASTError();
}

ArithmeticExprP Parser::ID() {
  matchToken(TokenType::Id);
  return std::make_unique<const Variable>(tokens[head].stringValue());
}

ArithmeticExprP Parser::parse() {
  return E();
}
}  // namespace cs160::frontend
