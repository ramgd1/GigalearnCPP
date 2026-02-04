#pragma once

#include "../Framework.h"

#include <nlohmann/json.hpp>

#include <iomanip>
#include <sstream>

namespace GGL {
	namespace Utils {
		template <typename T>
		nlohmann::json MakeJSONArray(const std::vector<T> list) {
			auto result = nlohmann::json::array();

			for (T v : list) {
				if (isnan(v))
					RG_LOG("MakeJSONArray(): Failed to serialize JSON with NAN value (list size: " << list.size() << ")");
				result.push_back(v);
			}

			return result;
		}

		template <typename T>
		std::vector<T> MakeVecFromJSON(const nlohmann::json& json) {
			return json.get<std::vector<T>>();
		}

		std::set<int64_t> FindNumberedDirs(std::filesystem::path basePath);

		template <typename T>
		std::string NumToStr(T val) {
			std::stringstream stream;
			auto addThousandsSep = [](std::string str) {
				if (str.empty())
					return str;

				char sign = 0;
				if (str[0] == '-' || str[0] == '+') {
					sign = str[0];
					str.erase(str.begin());
				}

				std::string exponentPart;
				size_t expPos = str.find_first_of("eE");
				if (expPos != std::string::npos) {
					exponentPart = str.substr(expPos);
					str = str.substr(0, expPos);
				}

				size_t dotPos = str.find('.');
				std::string intPart = (dotPos == std::string::npos) ? str : str.substr(0, dotPos);
				std::string fracPart = (dotPos == std::string::npos) ? "" : str.substr(dotPos);

				for (int i = (int)intPart.size() - 3; i > 0; i -= 3)
					intPart.insert(i, ",");

				std::string out = intPart + fracPart;
				if (sign)
					out.insert(out.begin(), sign);
				return out + exponentPart;
			};

			T valMag = val;
			if constexpr (std::is_signed<T>())
				valMag = abs(val);

			if ((valMag < 1e-3 && valMag > 0) || valMag >= 1e11) {
				stream << std::scientific << val;
			} else {
				if (val == (int64_t)val) {
					stream << (int64_t)val;
				} else {
					stream << std::fixed << std::setprecision(4) << val;
				}
			}

			std::string result = stream.str();
			if ((valMag < 1e-3 && valMag > 0) || valMag >= 1e11)
				return result;
			return addThousandsSep(result);
		}
	}
}
