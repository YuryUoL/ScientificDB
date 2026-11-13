#pragma once
#include <string>
#include <iostream>
#include <mysql.h>

class DatabaseConnection {
public:
    DatabaseConnection() {
        conn = mysql_init(nullptr);
    }

    ~DatabaseConnection() {
        disconnect();
    }

    bool connect(const std::string& host, const std::string& user, 
                 const std::string& password, const std::string& db, unsigned int port = 3306) 
    {
        if (!mysql_real_connect(conn, host.c_str(), user.c_str(), password.c_str(), db.c_str(), port, nullptr, 0)) {
            std::cerr << "[Database] Connection Error: " << mysql_error(conn) << "\n";
            connected = false;
            return false;
        }
        std::cout << "[Database] Successfully connected to MariaDB server at " << host << "\n";
        connected = true;
        return true;
    }

    void disconnect() {
        if (connected && conn) {
            mysql_close(conn);
            conn = nullptr;
            connected = false;
            std::cout << "[Database] Disconnected from MariaDB.\n";
        }
    }

    MYSQL* getHandle() { return conn; }
    bool isConnected() const { return connected; }

private:
    MYSQL* conn = nullptr;
    bool connected = false;
};