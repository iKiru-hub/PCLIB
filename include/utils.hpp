#pragma once
#include <iostream>



std::string get_datetime() {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%X", &tstruct);
    return buf;
} // get_datetime_str


namespace utils {


void logger(const std::string &msg,
            const std::string &src = "MAIN") {
    std::cout << get_datetime() << " | " << src \
        << " | " << msg << std::endl;
}

}
