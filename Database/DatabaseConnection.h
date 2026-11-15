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
                 const std::string& password, const std::string& db, unsigned int port = 3307) 
    {
        std::string socketPath = "/home/yury/Projects/CPPProj/MariaDB/data/mariadb.sock";
        
        bool reconnect = true;
        mysql_options(conn, MYSQL_OPT_RECONNECT, &reconnect);

        // Treat empty strings, literal quotes ("" or ''), as a null password
        bool isPassEmpty = password.empty() || password == "\"\"" || password == "''";
        const char* pwd_ptr = isPassEmpty ? nullptr : password.c_str();
        const char* db_ptr = (db.empty() || db == "\"\"" || db == "''") ? nullptr : db.c_str();

        if (!mysql_real_connect(conn, 
                                host.c_str(), 
                                user.c_str(), 
                                pwd_ptr, 
                                db_ptr, 
                                port, 
                                socketPath.c_str(), 
                                0)) 
        {
            std::cerr << "[Database] Connection Error: " << mysql_error(conn) << "\n";
            connected = false;
            return false;
        }

        std::cout << "[Database] Successfully connected to MariaDB server via socket.\n";
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