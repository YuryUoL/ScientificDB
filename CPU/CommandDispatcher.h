#pragma once

#include <string>
#include <sstream>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <vector>
#include <iomanip>

class CommandDispatcher {
public:
    using CommandHandler = std::function<void(std::istringstream&)>;

    // Registers a new command with explicit usage, short description, and optional detailed help block
    void registerCommand(const std::string& name, 
                         const std::string& usageSyntax, 
                         const std::string& helpDesc, 
                         CommandHandler handler,
                         const std::string& detailedHelp = "") {
        commands[name] = {usageSyntax, helpDesc, handler, detailedHelp};
        orderedNames.push_back(name);
    }

    // Maps an alias to an existing command
    void registerAlias(const std::string& alias, const std::string& targetCommand) {
        aliases[alias] = targetCommand;
    }

    // Parses raw CLI input line and routes to registered handler
    void execute(const std::string& inputLine) {
        std::istringstream iss(inputLine);
        std::string cmd;
        
        if (!(iss >> cmd)) {
            return; // Ignore empty lines
        }

        auto aliasIt = aliases.find(cmd);
        if (aliasIt != aliases.end()) {
            cmd = aliasIt->second;
        }

        auto it = commands.find(cmd);
        if (it != commands.end()) {
            it->second.handler(iss);
        } else {
            std::cout << "Unknown command: '" << cmd << "'. Type 'help' for a list of commands.\n";
        }
    }

    // Prints general overview OR detailed command-specific help
    void printHelp(const std::string& specificCmd = "") const {
        std::string target = specificCmd;
        auto aliasIt = aliases.find(target);
        if (aliasIt != aliases.end()) {
            target = aliasIt->second;
        }

        if (!target.empty()) {
            auto it = commands.find(target);
            if (it != commands.end()) {
                const auto& data = it->second;
                std::cout << "\n======================================================================\n"
                          << "                     COMMAND HELP: " << target << "\n"
                          << "======================================================================\n"
                          << "  Syntax: " << target << " " << data.usageSyntax << "\n"
                          << "  Description: " << data.helpDesc << "\n";
                if (!data.detailedHelp.empty()) {
                    std::cout << "\n" << data.detailedHelp << "\n";
                }
                std::cout << "======================================================================\n\n";
                return;
            } else {
                std::cout << "No help found for command: '" << specificCmd << "'. Type 'help' for the list.\n";
                return;
            }
        }

        // Default: Print all commands overview
        std::cout << "\n======================================================================\n";
        std::cout << "                         AVAILABLE COMMANDS                           \n";
        std::cout << "======================================================================\n";
        for (const auto& name : orderedNames) {
            const auto& data = commands.at(name);
            std::cout << "  " << std::left << std::setw(12) << name;
            if (!data.usageSyntax.empty()) {
                std::cout << " " << data.usageSyntax;
            }
            std::cout << "\n      - " << data.helpDesc << "\n\n";
        }
        std::cout << "  (Tip: Type 'help <command_name>' for detailed parameter options)\n";
        std::cout << "======================================================================\n\n";
    }

private:
    struct CommandData {
        std::string usageSyntax;
        std::string helpDesc;
        CommandHandler handler;
        std::string detailedHelp;
    };

    std::unordered_map<std::string, CommandData> commands;
    std::unordered_map<std::string, std::string> aliases;
    std::vector<std::string> orderedNames;
};