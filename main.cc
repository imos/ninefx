#include <glog/logging.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <future>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <vector>
using namespace std;

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName&); \
    void operator=(const TypeName&)

struct TException { string message; };
#define TCHECK(condition) for (;!(condition); throw TException()) \
    LOG(ERROR) << "Check failed: " #condition " "
#define TCHECK_OP(val1, val2, op) \
    for (;!((val1) op (val2)); throw TException()) \
    LOG(ERROR) << "Check failed: " #val1 " " #op " " #val2 " (" \
               << val1 << " vs " << val2 << ") "
#define TCHECK_EQ(val1, val2) TCHECK_OP(val1, val2, ==)
#define TCHECK_NE(val1, val2) TCHECK_OP(val1, val2, !=)
#define TCHECK_LE(val1, val2) TCHECK_OP(val1, val2, <=)
#define TCHECK_LT(val1, val2) TCHECK_OP(val1, val2, <)
#define TCHECK_GE(val1, val2) TCHECK_OP(val1, val2, >=)
#define TCHECK_GT(val1, val2) TCHECK_OP(val1, val2, >)
#define TCATCH(...) catch (TException&) \
    { LOG(ERROR) << "Catch failure: " << __VA_ARGS__; throw TException(); }

vector<pair<string, function<void()>>>* test_functions = nullptr;
int RegisterTestFunction(const string& name, function<void()> f) {
  if (test_functions == nullptr) {
    test_functions = new vector<pair<string, function<void()>>>();
  }
  test_functions->emplace_back(name, f);
  return test_functions->size();
}
void RunAllTests() {
  if (test_functions == nullptr) {
    LOG(ERROR) << "No tests to run.";
    return;
  }
  for (pair<string, function<void()>>& test_case : *test_functions) {
    LOG(INFO) << "Testing " << test_case.first << " ...";
    test_case.second();
  }
}
#define TEST(case_name) \
    void UnitTestFunction_ ## case_name (); \
    int unit_test_function_registry_ ## case_name = \
        RegisterTestFunction(#case_name, UnitTestFunction_ ## case_name); \
    void UnitTestFunction_ ## case_name ()

// 距離を計測するときに高値・安値を計算に入れるかどうか．0から1の間の値をとり，0の時は高値・
// 安値を計算に入れず，1の時は平均を値に入れません．（1が最良）
const double kHighAndLowDistanceWeight = 0;
// 予測に用いる過去の指標の割合（0.05程度が目安）
const double kPredictRatio = 0.01;

constexpr int kQFeatureSize = 1000;
constexpr int kQRedundancy = 50;

template<typename T>
int Sign(T x) { return x < 0 ? -1 : (x > 0 ? 1 : 0); }
bool IsNan(double value) { return ::std::isnan(value); }
bool IsInf(double value) { return ::std::isinf(value); }

string StringPrintf(const char* const format, ...) {
  va_list ap;
  va_start(ap, format);
  string result;
  for (int length = 100; length < 1024 * 1024; length *= 2) {
    char buf[length];
    va_list backup_ap;
    va_copy(backup_ap, ap);
    int output = vsnprintf(buf, length, format, backup_ap);
    va_end(backup_ap);

    if (output >= 0 && output < length) {
      result.append(buf, output);
      break;
    }
  }
  va_end(ap);
  return result;
}

TEST(StringPrintf) {
  CHECK_EQ("3.14", StringPrintf("%.2f", 3.1415));
}

vector<string> Split(const string& str, char delimiter) {
  istringstream iss(str); string tmp; vector<string> res;
  while (getline(iss, tmp, delimiter)) res.push_back(tmp);
  return res;
}

TEST(Split) {
  CHECK_EQ(0, Split("", 'x').size());

  {
    auto words = Split("foo", 'x');
    CHECK_EQ(1, words.size());
    CHECK_EQ("foo", words[0]);
  }

  {
    auto words = Split("abcxdefxghi", 'x');
    CHECK_EQ(3, words.size());
    CHECK_EQ("abc", words[0]);
    CHECK_EQ("def", words[1]);
    CHECK_EQ("ghi", words[2]);
  }
}

template<class T, class X = decltype(T().DebugString())>
ostream& operator<<(ostream& os, const T& value) {
  os << value.DebugString();
  return os;
}

template<typename T, typename InputType>
T CastWithBoundaryCheck(InputType value) {
  TCHECK(!IsInf(value));
  TCHECK(!IsNan(value));
  TCHECK_GE(value, static_cast<double>(numeric_limits<T>::min()));
  TCHECK_LE(value, static_cast<double>(numeric_limits<T>::max()));
  if (is_floating_point<InputType>::value) {
    return static_cast<T>(round(value));
  }
  return static_cast<T>(value);
}

class EnumType {
 public:
  EnumType(const string& candidates) {
    const char* from = candidates.c_str();
    for (int index = 0; *from != '\0'; index++) {
      while (isspace(*from) || *from == ',') { from++; }
      const char* to = from;
      while (*to != '\0' && !isspace(*to) && *to != ',') { to++; }
      names_.push_back(string(from, static_cast<size_t>(to - from)));
      from = to;
    }
  }

  int ParseEnvironmentInternal(const string& key) {
    char* value = getenv(key.c_str());
    if (value == nullptr) {
      return 0;
    }
    for (size_t i = 0; i < names_.size(); i++) {
      if (names_[i] == value) {
        return static_cast<int>(i);
      }
    }
    fprintf(stderr, "Invalid environment for %s: %s\n", key.c_str(), value);
    exit(1);
  }

  const string& GetNameInternal(size_t value) {
    return names_[value];
  }

 private:
  vector<string> names_;

  DISALLOW_COPY_AND_ASSIGN(EnumType);
};

#define REGISTER_ENUM(TypeName, ...)                                           \
    struct TypeName : public EnumType {                                        \
      TypeName() : EnumType(#__VA_ARGS__) {}                                   \
      enum Type { __VA_ARGS__ };                                               \
      Type ParseEnvironment(const string& key)                                 \
          { return static_cast<Type>(ParseEnvironmentInternal(key)); }         \
      const string& GetName(Type value) { return GetNameInternal(value); }     \
    }                                                                          \
    // マクロの終端

struct Params {
 public:
  REGISTER_ENUM(Mode, SIMULATE, EVALUATE, QLEARN);
  REGISTER_ENUM(Future, CLOSE, CROSS, LIMIT);
  REGISTER_ENUM(FutureCurve, FLAT, SQRT, LINEAR, LINEAR_CUT);

  Params() : initialized_(false) {}

  static void Init() {
    Params* params = MutableParams();
    params->num_threads = GetInteger("FX_NUM_THREADS", 32);
    params->mode = Mode().ParseEnvironment("FX_MODE");
    params->future = Future().ParseEnvironment("FX_FUTURE");
    params->future_curve = FutureCurve().ParseEnvironment("FX_FUTURE_CURVE");
    params->daily_volatility = (GetInteger("FX_DAILY_VOLATILITY", 0) != 0);
    params->base_volatility_interval =
        GetInteger("FX_BASE_VOLATILITY_INTERVAL", 24 * 60);
    params->future_variation = GetBoolean("FX_FUTURE_VARIATION", false);
    params->spread = GetFloat("FX_SPREAD", 0.29 / 10000);
    params->volatility_adjustment =
        (GetInteger("FX_VOLATILITY_ADJUSTMENT", 0) != 0);
    params->volatility_power = GetInteger("FX_VOLATILITY_POWER", 1);

    params->initialized_ = true;
  }

  static int GetInteger(const char* key, int default_value = 0) {
    char* value = getenv(key);
    if (value == nullptr) {
      return default_value;
    }
    return atoi(value);
  }

  static float GetFloat(const char* key, float default_value = 0) {
    char* value = getenv(key);
    if (value == nullptr) {
      return default_value;
    }
    return atof(value);
  }

  static bool GetBoolean(const char* key, bool default_value = false) {
    return GetInteger(key, default_value ? 1 : 0) != 0;
  }

  static string GetString(const char* key, const string& default_value = "") {
    char* value = getenv(key);
    if (value == nullptr) {
      return default_value;
    }
    return value;
  }

  static const Params& GetParams() {
    CHECK(MutableParams()->initialized_);
    return *MutableParams();
  }

  string DebugString(const string& indent = "") const {
    string result = "{\n";
    for (const pair<string, string>& key_to_value : GetJsonParameters()) {
      result += indent + "  \"" + key_to_value.first +
                "\": \"" + key_to_value.second + "\",\n";
    }
    result += indent + "}";
    return result;
  }

  string ShortDebugString() const {
    string result = "{";
    bool is_first = true;
    for (const pair<string, string>& key_to_value : GetJsonParameters()) {
      if (!is_first) { result += ","; }
      result += "\"" + key_to_value.first +
                "\":\"" + key_to_value.second + "\"";
      is_first = false;
    }
    result += "}";
    return result;
  }

  int num_threads;
  Mode::Type mode;
  Future::Type future;
  FutureCurve::Type future_curve;
  bool daily_volatility;
  int base_volatility_interval;
  bool future_variation;
  float spread;
  bool volatility_adjustment;
  int volatility_power;

 private:
  map<string, string> GetJsonParameters() const {
    return map<string, string>({
        {"num_threads", StringPrintf("%d", num_threads)},
        {"mode", Mode().GetName(mode)},
        {"future", Future().GetName(future)},
        {"future_curve", FutureCurve().GetName(future_curve)},
        {"volatility_adjustment", volatility_adjustment ? "true" : "false"},
        {"volatility_power", StringPrintf("%d", volatility_power)},
    });
  }

  static Params* MutableParams() {
    static Params params;
    return &params;
  }

  bool initialized_;

  DISALLOW_COPY_AND_ASSIGN(Params);
};

const Params& GetParams() { return Params::GetParams(); }

// 周期が2^64のXorshift乱数生成関数
uint32_t Rand() {
  static uint64_t x =
      88172645463325252ULL ^ static_cast<uint64_t>(
          Params::GetInteger("FX_RANDOM_SEED", time(nullptr)));
  x = x ^ (x << 13); x = x ^ (x >> 7);
  return static_cast<uint32_t>(x = x ^ (x << 17));
}

void Parallel(const function<bool(int)>& f) {
  if (GetParams().num_threads == 0) {
    while (f(0));
    return;
  }

  vector<thread> threads;
  for (int thread_id = 0; thread_id < GetParams().num_threads; thread_id++) {
    threads.push_back(thread([&f, thread_id]{ while (f(thread_id)); }));
  }
  for (thread& t : threads) { t.join(); }
}

template<class T, class S = T>
void ParallelFor(
    T begin, T end, S step, S chunk_step, const function<void(T)>& f) {
  T next = begin;
  mutex t_mutex;
  Parallel([&](int) {
    T base;
    {
      lock_guard<mutex> t_mutex_guard(t_mutex);
      base = next;
      if (!(base < end)) { return false; }
      next += chunk_step;
    }

    for (T current = base; current < end && current < base + chunk_step;
         current += step) {
      f(current);
    }
    return true;
  });
}

#define CLEAR_LINE "\033[1A\033[2K"

template<typename T, typename X, X F(const X&, const X&)>
class SegmentTree {
 public:
  SegmentTree() {}
  explicit SegmentTree(const vector<T>& data) { Init(data); }

  void Init(const vector<T>& data) {
    data_.clear();
    for (data_.push_back(data); data_.rbegin()->size() >= 2;) {
      const vector<T>& last_data = *data_.rbegin();
      vector<T> new_data;
      for (size_t i = 0; i * 2 + 1 < last_data.size(); i++) {
        new_data.push_back(F(last_data[i * 2], last_data[i * 2 + 1]));
      }
      data_.push_back(move(new_data));
    }
  }

  T Query(int from, int to) const {
    CHECK_LE(0, from);
    CHECK_LT(from, data_[0].size());
    CHECK_LE(0, to);
    CHECK_LT(to, data_[0].size());
    if (from > to) { return Query(to, from); }
    return InternalQuery(from, to, 0);
  }

  static void Test() {
    fprintf(stderr, "Testing SegmentTree...\n");

    SegmentTree<T, const T&, min<T>> small({1, 2, 3});
    CHECK_EQ(1, small.Query(0, 0));
    CHECK_EQ(2, small.Query(1, 1));
    CHECK_EQ(3, small.Query(2, 2));
    CHECK_EQ(1, small.Query(0, 1));
    CHECK_EQ(2, small.Query(1, 2));
    CHECK_EQ(1, small.Query(0, 2));

    vector<int32_t> data;
    for (int i = 0; i < 29; i++) {
      data.push_back(Rand() % 100);
    }
    SegmentTree segment_tree(data);
    for (int i = 0; i < (int)data.size(); i++) {
      for (int j = i; j < (int)data.size(); j++) {
        CHECK_EQ(segment_tree.Query(i, j), segment_tree.Naive(i, j))
            << "i=" << i << ", j=" << j;
      }
    }

    fprintf(stderr, "SegmentTree test successfully passed.\n");
  }

 private:
  T InternalQuery(int from, int to, size_t depth) const {
    assert(0 <= depth && depth < data_.size());
    const vector<T>& data = data_[depth];

    assert(0 <= from && from < (int)data.size());
    assert(0 <= to && to < (int)data.size());
    assert(from <= to);
    if (from == to) { return data[(size_t)from]; }
    if (from % 2 != 0) {
      return F(data[(size_t)from], InternalQuery(from + 1, to, depth));
    }
    if (to % 2 == 0) {
      return F(data[(size_t)to], InternalQuery(from, to - 1, depth));
    }
    return InternalQuery(from / 2, to / 2, depth + 1);
  }

  T Naive(int from, int to) const {
    assert(0 <= from && from < (int)data_[0].size());
    assert(0 <= to && to < (int)data_[0].size());
    if (from > to) { return Naive(to, from); }
    T result = data_[0][(size_t)from];
    for (int i = from; i <= to; i++) {
      result = F(result, data_[0][(size_t)i]);
    }
    return result;
  }

  vector<vector<T>> data_;

  DISALLOW_COPY_AND_ASSIGN(SegmentTree);
};

template<typename FinalType, typename ValueType = int32_t>
struct ScalarBase {
  static constexpr ValueType kInvalid = numeric_limits<ValueType>::max();

  ScalarBase() : value_(0) {}
  explicit ScalarBase(ValueType value) : value_(value) {}

  int64_t GetRawValue() const
      { CHECK(IsValid()) << StaticDebugString(); return value_; }
  template<class InputType>
  void SetRawValue(InputType value)
      { value_ = CastWithBoundaryCheck<ValueType>(value); }

  void Clear() { *this = FinalType(); }

  bool IsValid() const { return value_ != kInvalid; }
  void Invalidate() { value_ = kInvalid; }

  #define SCALAR_BASE_COMPARISON_OPERATOR(Operator)                            \
      bool operator Operator(FinalType value) const {                          \
        return IsValid() && value.IsValid() &&                                 \
               GetRawValue() Operator value.GetRawValue();                     \
      }
  SCALAR_BASE_COMPARISON_OPERATOR(<);
  SCALAR_BASE_COMPARISON_OPERATOR(<=);
  SCALAR_BASE_COMPARISON_OPERATOR(>);
  SCALAR_BASE_COMPARISON_OPERATOR(>=);
  SCALAR_BASE_COMPARISON_OPERATOR(==);
  SCALAR_BASE_COMPARISON_OPERATOR(!=);

  static FinalType Invalid()
      { FinalType result; result.Invalidate(); return result; }

  template<class InputType>
  static FinalType InRawValue(InputType value)
      { FinalType result; result.SetRawValue(value); return result; }

  static string StaticDebugString() { return typeid(FinalType).name(); }

 protected:
  const FinalType& FinalValue() const
      { return static_cast<const FinalType&>(*this); }

 private:
  ValueType value_;

  // virtualなどの関数を作らせないための制限を設定
  // static_assert(is_trivially_copyable<FinalType>::value,
  //               "A scalar value must be trivially copyable.");
};

template<typename FinalType, typename ValueType = int32_t>
struct AdditiveScalarBase : public ScalarBase<FinalType, ValueType> {
  typedef double WeightType;

  AdditiveScalarBase() : ScalarBase<FinalType, ValueType>()
      { static_cast<FinalType*>(this)->SetWeight(0); }
  explicit AdditiveScalarBase(ValueType value, WeightType weight)
      : ScalarBase<FinalType, ValueType>(value)
      { static_cast<FinalType*>(this)->SetWeight(weight); }

  const FinalType& operator-=(FinalType value) {
    this->SetRawValue(this->GetRawValue() - value.GetRawValue());
    static_cast<FinalType*>(this)->SetWeight(
        static_cast<FinalType*>(this)->GetWeight() - value.GetWeight());
    return this->FinalValue();
  }
  FinalType operator-() const
      { FinalType result; return result -= this->FinalValue(); }
  FinalType operator-(FinalType value) const
      { FinalType result = this->FinalValue(); return result -= value; }
  FinalType operator+=(FinalType value)
      { return static_cast<FinalType&>(*this) -= -value; }
  FinalType operator+(FinalType value) const
      { return static_cast<const FinalType&>(*this) - -value; }

  const FinalType& operator*=(double ratio) {
    this->SetRawValue(this->GetRawValue() * ratio);
    static_cast<FinalType*>(this)->SetWeight(
        static_cast<FinalType*>(this)->GetWeight() * ratio);
    return this->FinalValue();
  }
  FinalType operator*(double ratio) const
      { FinalType result = this->FinalValue(); return result *= ratio; }
  const FinalType& operator/=(double ratio) { return *this *= 1 / ratio; }
  FinalType operator/(double ratio) const
      { FinalType result = this->FinalValue(); return result /= ratio; }

 protected:
  int32_t GetWeight() const { return 0; }
  void SetWeight(double) {}
};

template<typename FinalType, typename ValueType = int32_t>
struct AdditiveScalarWithWeightBase
    : public AdditiveScalarBase<FinalType, ValueType> {
  AdditiveScalarWithWeightBase() : AdditiveScalarBase<FinalType, ValueType>() {}
  explicit AdditiveScalarWithWeightBase(ValueType value, double weight)
      : AdditiveScalarBase<FinalType, ValueType>(value, weight) {}

  double GetWeight() const { return weight_; }
  void SetWeight(double weight) { weight_ = weight; }

 private:
  double weight_;
};

#ifndef NDEBUG
#define AdditiveScalarWithoutWeightBase AdditiveScalarWithWeightBase
#else
#define AdditiveScalarWithoutWeightBase AdditiveScalarBase
#endif

template<typename RootType>
struct DifferenceBase
    : public AdditiveScalarWithoutWeightBase<
          typename RootType::DifferenceType, typename RootType::ScalarType> {
  typedef AdditiveScalarWithoutWeightBase<
              typename RootType::DifferenceType, typename RootType::ScalarType>
          ParentType;

  DifferenceBase() : ParentType() {}
};

template<typename RootType>
struct ValueBase
    : public ScalarBase<typename RootType::ValueType,
                        typename RootType::ScalarType> {
  typedef ScalarBase<typename RootType::ValueType,
                     typename RootType::ScalarType> ParentType;
  typedef typename RootType::DifferenceType DifferenceType;
  typedef typename RootType::ValueType ValueType;

  ValueBase() : ParentType() { this->Invalidate(); }

  DifferenceType operator-(ValueType value) const {
    return DifferenceType::InRawValue(
        this->GetRawValue() - value.GetRawValue());
  }
  const ValueType& operator+=(DifferenceType value) {
    this->SetRawValue(this->GetRawValue() + value.GetRawValue());
    return this->FinalValue();
  }
  const ValueType& operator-=(DifferenceType value) { return *this += -value; }
  ValueType operator+(DifferenceType value) const
      { ValueType result = this->FinalValue(); return result += value; }
  ValueType operator-(DifferenceType value) const
      { ValueType result = this->FinalValue(); return result -= value; }

 private:
  const ValueType& FinalValue() const
      { return static_cast<const ValueType&>(*this); }
};

template<typename RootType>
struct SumBase : public AdditiveScalarWithoutWeightBase<
                            typename RootType::SumType, int64_t> {
  typedef typename RootType::ValueType ValueType;
  typedef typename RootType::SumType SumType;

  SumBase() : AdditiveScalarWithoutWeightBase<
                  typename RootType::SumType, int64_t>() {}

  ValueType GetAverage(double weight) const {
    CHECK(!IsNan(weight));
    CHECK(!IsInf(weight));
    CHECK(fabs(weight) > 1e-6);
#ifndef NDEBUG
    CHECK_NEAR(this->GetWeight(), weight, 1e-6);
#endif
    return ValueType::InRawValue(this->GetRawValue() / weight);
  }

  // SegmentTree用の加算関数．
  // NOTE: SegmentTreeの関数はconst参照渡しのみ対応．
  static SumType Add(const SumType& a, const SumType& b) { return a + b; }

  static SumType From(ValueType value)
      { return InRawValue(value.GetRawValue(), 1); }

  template<class InputType>
  static SumType InRawValue(InputType value, double weight) {
    SumType result;
    result.SetRawValue(value);
    result.SetWeight(weight);
    return result;
  }
};

struct TimeRoot {
  typedef int32_t ScalarType;

  struct DifferenceType : public DifferenceBase<TimeRoot> {
    DifferenceType() : DifferenceBase<TimeRoot>() {}

    int32_t GetSecond() const { return GetRawValue(); }
    double GetMinute() const { return GetSecond() / 60.0; }
    string DebugString() const
        { return StringPrintf("%.1f minute(s)", GetMinute()); }
    static DifferenceType InMinute(double minute)
        { return DifferenceType::InRawValue(minute * 60); }
  };

  struct ValueType : public ValueBase<TimeRoot> {
    ValueType() : ValueBase<TimeRoot>() {}

    bool Load(FILE* fp) {
      int32_t minute;
      if (fread(&minute, sizeof(minute), 1, fp) <= 0) { return false; }
      SetRawValue(static_cast<int64_t>(minute) * 60);
      return true;
    }

    int32_t GetSecond() const { return GetRawValue(); }
    double GetMinute() const { return GetSecond() / 60.0; }

    // TODO(imos): Deprecate this.
    void AddMinute(int32_t minute) {
      *this += DifferenceType::InMinute(minute);
    }

    // 1970年1月1日0時(UTC)から経過した時間を分単位で返します．
    int32_t GetMinuteIndex() const {
      return static_cast<int32_t>((GetSecond() + 30) / 60);
    }

    int32_t GetWeekIndex() const {
      return (GetMinuteIndex() - 4320) / (60 * 24 * 7);
    }

    // 直近の日曜日0時(UTC)から経過した時間を分単位で返します．
    int32_t GetWeeklyIndex() const {
      return (GetMinuteIndex() - 4320) % (60 * 24 * 7);
    }

    int GetIndex(ValueType base_time) const {
      return GetMinuteIndex() - base_time.GetMinuteIndex();
    }

    // 計算に安全な時刻であるかを返す．
    bool IsSafeTime(bool include_friday) const {
      int weekly_index = GetWeeklyIndex();
      // 月曜日の1時(UTC)までは週末の影響を受けやすいので除去．
      if (weekly_index < 60 * 24 * 1 + 60) { return false; }
      // 金曜日の18時(UTC)以降は流動性が下がるので除去．
      if (weekly_index >= 60 * 24 * 5 + 60 * 18) { return false; }
      // [OPTIONAL] 金曜日の9時(UTC)以降は指標発表の影響を受けるので除去．
      if (include_friday &&
          weekly_index >= 60 * 24 * 5 + 60 * 9) { return false; }
      return true;
    }

    bool IsSummerTime() const {
      CHECK(IsValid());
      int64_t t = GetRawValue();
      if (954658800 < t && t < 972799200) { return true; } // 2000
      if (986108400 < t && t < 1004248800) { return true; } // 2001
      if (1018162800 < t && t < 1035698400) { return true; } // 2002
      if (1049612400 < t && t < 1067148000) { return true; } // 2003
      if (1081062000 < t && t < 1099202400) { return true; } // 2004
      if (1112511600 < t && t < 1130652000) { return true; } // 2005
      if (1143961200 < t && t < 1162101600) { return true; } // 2006
      if (1173596400 < t && t < 1194156000) { return true; } // 2007
      if (1205046000 < t && t < 1225605600) { return true; } // 2008
      if (1236495600 < t && t < 1257055200) { return true; } // 2009
      if (1268550000 < t && t < 1289109600) { return true; } // 2010
      if (1299999600 < t && t < 1320559200) { return true; } // 2011
      if (1331449200 < t && t < 1352008800) { return true; } // 2012
      if (1362898800 < t && t < 1383458400) { return true; } // 2013
      if (1394348400 < t && t < 1414908000) { return true; } // 2014
      if (1425798000 < t && t < 1446357600) { return true; } // 2015
      if (1457852400 < t && t < 1478412000) { return true; } // 2016
      if (1489302000 < t && t < 1509861600) { return true; } // 2017
      if (1520751600 < t && t < 1541311200) { return true; } // 2018
      if (1552201200 < t && t < 1572760800) { return true; } // 2019
      if (1583650800 < t && t < 1604210400) { return true; } // 2020
      if (1615705200 < t && t < 1636264800) { return true; } // 2021
      if (1647154800 < t && t < 1667714400) { return true; } // 2022
      if (1678604400 < t && t < 1699164000) { return true; } // 2023
      if (1710054000 < t && t < 1730613600) { return true; } // 2024
      if (1741503600 < t && t < 1762063200) { return true; } // 2025
      if (1772953200 < t && t < 1793512800) { return true; } // 2026
      if (1805007600 < t && t < 1825567200) { return true; } // 2027
      if (1836457200 < t && t < 1857016800) { return true; } // 2028
      if (1867906800 < t && t < 1888466400) { return true; } // 2029
      if (1899356400 < t && t < 1919916000) { return true; } // 2030
      if (1930806000 < t && t < 1951365600) { return true; } // 2031
      if (1962860400 < t && t < 1983420000) { return true; } // 2032
      if (1994310000 < t && t < 2014869600) { return true; } // 2033
      if (2025759600 < t && t < 2046319200) { return true; } // 2034
      if (2057209200 < t && t < 2077768800) { return true; } // 2035
      if (2088658800 < t && t < 2109218400) { return true; } // 2036
      if (2120108400 < t && t < 2140668000) { return true; } // 2037
      return false;
    }

    string DebugString() const {
      if (!IsValid()) { return "\?\?\?\?-\?\?-\?\? \?\?:\?\?:\?\?"; }
      time_t t = static_cast<time_t>(GetRawValue());
      struct tm *utc = gmtime(&t);
      return StringPrintf(
          "%04d-%02d-%02d %02d:%02d:%02d",
          utc->tm_year + 1900, utc->tm_mon + 1, utc->tm_mday,
          utc->tm_hour, utc->tm_min, utc->tm_sec);
    }

    static ValueType InMinute(double minute)
        { return ValueType::InRawValue(minute * 60); }

    // 指定された年の1月1日の時刻を返します．
    static ValueType InYear(int32_t year) {
      if (year == 2000) { return ValueType::InRawValue(946652400); }
      if (year == 2001) { return ValueType::InRawValue(978274800); }
      if (year == 2002) { return ValueType::InRawValue(1009810800); }
      if (year == 2003) { return ValueType::InRawValue(1041346800); }
      if (year == 2004) { return ValueType::InRawValue(1072882800); }
      if (year == 2005) { return ValueType::InRawValue(1104505200); }
      if (year == 2006) { return ValueType::InRawValue(1136041200); }
      if (year == 2007) { return ValueType::InRawValue(1167577200); }
      if (year == 2008) { return ValueType::InRawValue(1199113200); }
      if (year == 2009) { return ValueType::InRawValue(1230735600); }
      if (year == 2010) { return ValueType::InRawValue(1262271600); }
      if (year == 2011) { return ValueType::InRawValue(1293807600); }
      if (year == 2012) { return ValueType::InRawValue(1325343600); }
      if (year == 2013) { return ValueType::InRawValue(1356966000); }
      if (year == 2014) { return ValueType::InRawValue(1388502000); }
      if (year == 2015) { return ValueType::InRawValue(1420038000); }
      if (year == 2016) { return ValueType::InRawValue(1451574000); }
      if (year == 2017) { return ValueType::InRawValue(1483196400); }
      if (year == 2018) { return ValueType::InRawValue(1514732400); }
      if (year == 2019) { return ValueType::InRawValue(1546268400); }
      if (year == 2020) { return ValueType::InRawValue(1577804400); }
      if (year == 2021) { return ValueType::InRawValue(1609426800); }
      if (year == 2022) { return ValueType::InRawValue(1640962800); }
      if (year == 2023) { return ValueType::InRawValue(1672498800); }
      if (year == 2024) { return ValueType::InRawValue(1704034800); }
      if (year == 2025) { return ValueType::InRawValue(1735657200); }
      if (year == 2026) { return ValueType::InRawValue(1767193200); }
      if (year == 2027) { return ValueType::InRawValue(1798729200); }
      if (year == 2028) { return ValueType::InRawValue(1830265200); }
      if (year == 2029) { return ValueType::InRawValue(1861887600); }
      if (year == 2030) { return ValueType::InRawValue(1893423600); }
      if (year == 2031) { return ValueType::InRawValue(1924959600); }
      if (year == 2032) { return ValueType::InRawValue(1956495600); }
      if (year == 2033) { return ValueType::InRawValue(1988118000); }
      if (year == 2034) { return ValueType::InRawValue(2019654000); }
      if (year == 2035) { return ValueType::InRawValue(2051190000); }
      if (year == 2036) { return ValueType::InRawValue(2082726000); }
      if (year == 2037) { return ValueType::InRawValue(2114348400); }
      LOG(FATAL) << "Unsupported year: " << year;
      return ValueType();
    }
  };

  struct SumType : public SumBase<TimeRoot> {
    SumType() : SumBase<TimeRoot>() {}

    bool IsValid() const { return GetRawValue() != kInvalid; }
  };
};

typedef typename TimeRoot::DifferenceType TimeDifference;
typedef typename TimeRoot::ValueType Time;
typedef typename TimeRoot::SumType TimeSum;

struct PriceRoot {
  typedef int32_t ScalarType;

  static constexpr double kLogPriceRatio = 1.0e8;

  struct DifferenceType : public DifferenceBase<PriceRoot> {
    DifferenceType() : DifferenceBase<PriceRoot>() {}

    // 価格比の自然対数で設定・取得を行います．
    double GetLogValue() const { return GetRawValue() / kLogPriceRatio; }
    void SetLogValue(double log_price)
        { SetRawValue(log_price * kLogPriceRatio); }

    string DebugString() const { return StringPrintf("%+.4f", GetLogValue()); }

    static DifferenceType InRatio(double ratio) {
      return InRawValue(log(ratio) * kLogPriceRatio);
    }
  };

  struct ValueType : public ValueBase<PriceRoot> {
    ValueType() : ValueBase<PriceRoot>() {}

    bool Load(FILE* fp) {
      int32_t price;
      if (fread(&price, sizeof(price), 1, fp) <= 0) { return false; }
      SetRawValue(price);
      return true;
    }

    // TODO(imos): Fix this.
    int32_t GetLogPrice() const { return GetRawValue(); }
    void SetLogPrice(double log_price) { SetRawValue(log_price); }

    double GetRealPrice() const { return exp(GetRawValue() / kLogPriceRatio); }
    void SetRealPrice(double real_price)
        { SetRawValue(log(real_price) * kLogPriceRatio); }

    string DebugString() const {
      if (!IsValid()) { return "NaN"; }
      double real_price = GetRealPrice();
      int upper_digits = max((int)floor(log10(real_price) + 1), 1);
      int lower_digits = max(0, 6 - upper_digits);
      return StringPrintf("%.*f", lower_digits, real_price);
    }

    static ValueType InRealPrice(double real_price) {
      ValueType result; result.SetRealPrice(real_price); return result;
    }
  };

  struct SumType : public SumBase<PriceRoot> {
    SumType() : SumBase<PriceRoot>() {}
  };
};

typedef typename PriceRoot::DifferenceType PriceDifference;
typedef typename PriceRoot::ValueType Price;
typedef typename PriceRoot::SumType PriceSum;

TEST(Price) {
  PriceDifference d;
  CHECK(d.IsValid());
  CHECK_EQ(d.GetLogValue(), 0);
  d.SetLogValue(0.01);
  CHECK_NEAR(d.GetLogValue(), 0.01, 1e-6);

  CHECK_NEAR(PriceDifference::InRatio(1.001).GetLogValue(), 0.001, 1e-6);
  CHECK_NEAR(PriceDifference::InRatio(0.999).GetLogValue(), -0.001, 1e-6);
  CHECK_NEAR((PriceDifference::InRatio(1.001) +
              PriceDifference::InRatio(1.002)).GetLogValue(),
             0.003, 1e-5);
  CHECK_NEAR((PriceDifference::InRatio(1.001) -
              PriceDifference::InRatio(1.002)).GetLogValue(),
             -0.001, 1e-5);

  Price p;
  CHECK(!p.IsValid());
  p.SetRealPrice(123.45);
  CHECK_NEAR(p.GetRealPrice(), 123.45, 1e-6);
  p += PriceDifference::InRatio(1.5);
  CHECK_NEAR(p.GetRealPrice(), 123.45 * 1.5, 1e-4);
  p -= PriceDifference::InRatio(1.5);
  CHECK_NEAR(p.GetRealPrice(), 123.45, 1e-6);

  CHECK_NEAR(Price::InRealPrice(100.01).GetRealPrice(), 100.01, 1e-6);
  CHECK_NEAR((Price::InRealPrice(100.02) -
              Price::InRealPrice(100.01)).GetLogValue(),
             log(100.02 / 100.01), 1e-8);
  CHECK_NEAR((Price::InRealPrice(100.01) -
              Price::InRealPrice(100.02)).GetLogValue(),
             -log(100.02 / 100.01), 1e-8);
}

// [start_time, end_time) の時間を 1 分単位でイテレートします．
class TimeIterator {
 public:
  TimeIterator(Time start_time, Time end_time, bool show_progress)
      : start_time_(start_time),
        end_time_(end_time),
        next_time_(start_time),
        show_progress_(show_progress) {
    CHECK(start_time_.IsValid());
    CHECK(end_time_.IsValid());
    CHECK_LE(start_time_, end_time_);
    if (show_progress_) {
      fprintf(stderr, "- Processing 0%%...\n");
    }
  }

  size_t Count() const {
    return end_time_.GetMinuteIndex() - start_time_.GetMinuteIndex();
  }

  int GetPercentage(Time time) const {
    return 100 * (time.GetMinuteIndex() - start_time_.GetMinuteIndex())
           / (end_time_.GetMinuteIndex() - start_time_.GetMinuteIndex());
  }

  Time GetAndIncrement() {
    Time time;
    {
      lock_guard<mutex> mutex_guard_(mutex);
      time = next_time_;
      if (!(time < end_time_)) {
        time = Time::Invalid();
      }
      next_time_.AddMinute(1);
    }
    if (show_progress_) {
      if (time.IsValid()) {
        int percentage = GetPercentage(time);
        if (percentage !=
            GetPercentage(time - TimeDifference::InMinute(1))) {
          fprintf(stderr,
                  CLEAR_LINE "- Processing %d%%...\n",
                  percentage);
        }
      } else {
        fprintf(stderr, CLEAR_LINE "Successfully processed.\n");
        show_progress_ = false;
      }
    }
    return time;
  }

 private:
  Time start_time_;
  Time end_time_;
  mutex mutex_;
  Time next_time_;
  bool show_progress_;

  DISALLOW_COPY_AND_ASSIGN(TimeIterator);
};

struct Asset {
 public:
  Asset() : currency_(1.0),
            foreign_currency_(0.0),
            trade_(0.0),
            total_fee_(0.0),
            highest_value_(1.0),
            drawdown_(0.0),
            position_value_(1.0),
            position_currency_(1.0),
            position_foreign_currency_(0.0) {}

  void UpdateStats(Time current_time, Price current_price) {
    CHECK(current_time.IsValid());
    CHECK(current_price.IsValid());
    if (!last_time_.IsValid()) { last_time_ = current_time; }

    highest_value_ = max(highest_value_, GetValue(current_price));
    drawdown_ = max(drawdown_, highest_value_ - GetValue(current_price));
    double leverage = GetLeverage(current_price);
    hold_ += (current_time - last_time_) * fabs(leverage);
    signed_hold_ += (current_time - last_time_) * leverage;
    position_value_ = position_currency_ +
                      current_price.GetRealPrice() * position_foreign_currency_;
    position_foreign_currency_ =
        position_value_ * fabs(leverage) / current_price.GetRealPrice();
    position_currency_ =
        position_value_ -
        position_foreign_currency_ * current_price.GetRealPrice();

    last_time_ = current_time;
  }

  void Trade(Time current_time,
             Price current_price,
             double leverage,
             bool use_spread = false) {
    CHECK(current_time.IsValid());
    CHECK(current_price.IsValid());

    double value = GetValue(current_price);
    double last_currency = currency_;
    foreign_currency_ = value * leverage / current_price.GetRealPrice();
    currency_ = value - foreign_currency_ * current_price.GetRealPrice();
    if (use_spread) {
      double fee = fabs(last_currency - currency_) * GetParams().spread / 2;
      currency_ -= fee;
      total_fee_ += fee;
    }
    trade_ += fabs(last_currency - currency_);
    UpdateStats(current_time, current_price);
  }

  double GetValue(Price current_price) const {
    return currency_ + current_price.GetRealPrice() * foreign_currency_;
  }

  double GetTotalFee() const { return total_fee_; }

  double GetTotalTrade() const { return trade_; }

  double GetLeverage(Price current_price) const {
    return 1 - currency_ / GetValue(current_price);
  }

  string Stats() const {
    return StringPrintf("max drawdown: %.1f%%, ", drawdown_ * 100) +
           StringPrintf("hold time: %.2f days, ",
                        hold_.GetMinute() / 60 / 24) +
           StringPrintf("leverage: %.2f, ",
                        signed_hold_.GetMinute() / hold_.GetMinute()) +
           StringPrintf("base value: %.6f, ", position_value_) +
           StringPrintf("trade: %.2f", trade_);
  }

 private:
  double currency_;
  double foreign_currency_;
  double trade_;
  double total_fee_;

  // 統計情報
  Time last_time_;
  double highest_value_;
  double drawdown_;
  TimeDifference hold_;
  TimeDifference signed_hold_;
  double position_value_;
  double position_currency_;
  double position_foreign_currency_;
};

struct VolatilityRatio {
 public:
  VolatilityRatio() : ratio_(1.0) {}
  explicit VolatilityRatio(double ratio) : ratio_((float)ratio) {}

  double GetValue() const { return ratio_; }

 private:
  float ratio_;
};

// 分あたりの変動量（差の2乗）を保持します
struct Volatility {
 public:
  // 2乗を保存するためのスケール
  // (e.g. 2 pips の変動は 20000^2 / kVolatilityScaleDeprecated として保存されます．)
  static constexpr double kVolatilityScale = 1000000;
  static const int32_t kVolatilityScaleDeprecated = 10000;

  Volatility() : volatility_(-1) {}

  VolatilityRatio operator/(Volatility value) const {
    return VolatilityRatio(
        (float)sqrt((double)volatility_ / value.volatility_));
  }

  Volatility operator*(VolatilityRatio ratio) const {
    return Volatility(
        GetValueDeprecated() * ratio.GetValue() * ratio.GetValue());
  }

  bool IsValid() const {
    return volatility_ >= 0;
  }

  void SetValue(double value) {
    value = pow(value * kVolatilityScale, 2.0);
    CHECK(!IsNan(value));
    CHECK(!IsInf(value));
    CHECK_GE(value, 0);
    CHECK_LE(value, numeric_limits<int32_t>::max());
    volatility_ = static_cast<int32_t>(round(value));
  }

  double GetValue() const {
    return sqrt(volatility_) / kVolatilityScale;
  }

  void SetValueDeprecated(double value) {
    SetValue(sqrt(value / kVolatilityScaleDeprecated) / kVolatilityScale);
    // CHECK(!IsNan(value));
    // CHECK(!IsInf(value));
    // CHECK_GE(value / kVolatilityScaleDeprecated,
    //          numeric_limits<int32_t>::min());
    // CHECK_LE(value / kVolatilityScaleDeprecated,
    //          numeric_limits<int32_t>::max());
    // CHECK_GE(value, 0);
    // volatility_ = (int32_t)round(value / kVolatilityScaleDeprecated);
  }

  double GetValueDeprecated() const {
    return pow(GetValue() * kVolatilityScale, 2) * kVolatilityScaleDeprecated;
    // return static_cast<double>(volatility_) * kVolatilityScaleDeprecated;
  }

  int32_t GetRawValue() const { return volatility_; }
  void SetRawValue(int64_t value) {
    assert(value >= numeric_limits<int32_t>::min());
    assert(value <= numeric_limits<int32_t>::max());
    volatility_ = static_cast<int32_t>(value);
  }

  Price GetPrice(Price current_price,
                 TimeDifference time_difference,
                 double ratio) const {
    PriceDifference price_difference;
    price_difference.SetLogValue(
        GetValue() * sqrt(time_difference.GetMinute()) * ratio);
    return current_price + price_difference;
  }

  string DebugString() const {
    return StringPrintf("%.6f%%", GetValue() * 100);
  }

  static Volatility Invalid() {
    return Volatility();
  }

  static Volatility InRatio(double ratio) {
    Volatility result;
    result.SetValue(ratio);
    return result;
  }

  static void Test() {
    fprintf(stderr, "Testing Volatility...\n");

    // ボラティリティ解像度を確認
    {
      Volatility volatility;
      volatility.SetRawValue(1);
      double ratio = volatility.GetPrice(Price::InRealPrice(1.0),
                                         TimeDifference::InMinute(60),
                                         1.0).GetRealPrice();
      fprintf(stderr, "- Minimal resolution: %.3e\n", ratio - 1);
      fprintf(stderr, "- Maximal resolution: %.3e\n",
              (ratio - 1) * sqrt(numeric_limits<int32_t>::max()));
      CHECK_LE(1, ratio);
      // 0.1pipsの差が表現可能であるか
      CHECK_LE(ratio, 1.00001) << "Volatility cannot represents 0.1 pips "
                               << "difference in 60 minutes.";
      // exp(10) が表現可能であるかどうか
      // TODO(imos): 表現できていないので要修正
      CHECK_GT((ratio - 1) * sqrt(numeric_limits<int32_t>::max()), 0.3);
    }

    // 入出力を確認
    {
      Volatility volatility;
      volatility.SetValue(0.001);
      double value = volatility.GetValue();
      CHECK(fabs(value - 0.001) < 1e-7) << "Inconsistent interface: " << value;
    }

    // 入出力を確認
    {
      Volatility volatility;
      volatility.SetValue(0.001);
      double ratio = volatility.GetPrice(Price::InRealPrice(1.0),
                                         TimeDifference::InMinute(1),
                                         1.0).GetRealPrice();
      CHECK(fabs(ratio - 1.001) < 1e-5) << "Inconsistent interface: " << ratio;
    }

    // 入出力を確認
    {
      Volatility volatility;
      volatility.SetValueDeprecated(10000);
      double value = volatility.GetValueDeprecated();
      CHECK(fabs(value - 10000) < 1e-3) << "Inconsistent interface: " << value;
    }

    // 後方互換性の確認
    {
      CHECK_EQ(1000000, Volatility::InRatio(0.001).GetRawValue());
      CHECK_EQ(4000000, Volatility::InRatio(0.002).GetRawValue());
    }
  }

 private:
  Volatility(double volatility)
      : volatility_((int32_t)round(volatility / kVolatilityScaleDeprecated)) {
    assert(volatility >= 0);
    assert(round(volatility / kVolatilityScaleDeprecated) <= 0x7fffffff);
  }

  int32_t volatility_;
};

struct VolatilitySum {
 public:
  VolatilitySum() : volatility_(0), count_(0) {}

  VolatilitySum(const Volatility& volatility)
      : volatility_(volatility.IsValid() ? volatility.GetRawValue() : 0),
        count_(volatility.IsValid() ? 1 : 0) {}

  Volatility GetAverageVolatility() const {
    assert(count_ >= 0);
    if (count_ == 0) {
      return Volatility::Invalid();
    }
    int64_t new_volatility = (volatility_ + count_ / 2) / count_;
    assert(0 <= new_volatility);
    assert(new_volatility < 0x7fffffff);
    Volatility result;
    result.SetRawValue((int32_t)new_volatility);
    return result;
  }

  VolatilitySum operator+(VolatilitySum x) const {
    return VolatilitySum(volatility_ + x.volatility_,
                         count_ + x.count_);
  }

  const VolatilitySum& operator+=(Volatility value) {
    if (value.IsValid()) {
      volatility_ += value.GetRawValue();
      assert(0 <= volatility_);
      count_++;
      assert(0 <= count_);
    }
    return *this;
  }

  const VolatilitySum& operator-=(Volatility value) {
    if (value.IsValid()) {
      volatility_ -= value.GetRawValue();
      assert(0 <= volatility_);
      count_--;
      assert(0 <= count_);
    }
    return *this;
  }

  int32_t GetCount() const { return count_; }

  // SegmentTree用の加算関数．
  // NOTE: SegmentTreeの関数はconst参照渡しのみ対応．
  static VolatilitySum Add(const VolatilitySum& a, const VolatilitySum& b) {
    return a + b;
  }

 private:
  VolatilitySum(int64_t volatility, int32_t count)
      : volatility_(volatility), count_(count) {
    assert(0 <= volatility_);
    assert(0 <= count_);
  }

  int64_t volatility_;
  int32_t count_;
};

struct Rate {
 public:
  bool IsValid() const {
    return time_.IsValid();
  }

  bool IsComplete() const {
    return time_.IsValid() && open_.IsValid() && high_.IsValid() &&
           low_.IsValid() && close_.IsValid();
  }

  bool Load(FILE* fp) {
    if (!time_.Load(fp)) { return false; }
    if (!open_.Load(fp)) { return false; }
    if (!high_.Load(fp)) { return false; }
    if (!low_.Load(fp)) { return false; }
    if (!close_.Load(fp)) { return false; }
    average_ = (PriceSum::From(open_) + PriceSum::From(high_) +
                PriceSum::From(low_) + PriceSum::From(close_)).GetAverage(4);
    return true;
  }

  string DebugString() const {
    return time_.DebugString()
        + " [open: " + open_.DebugString() + ", "
        + "high: " + high_.DebugString() + ", "
        + "low: " + low_.DebugString() + ", "
        + "close: " + close_.DebugString() + ", "
        + "avg: " + average_.DebugString() + "]";
  }

  inline Time GetTime() const { return time_; }
  inline Price GetOpenPrice() const { return open_; }
  inline Price GetHighPrice() const { return high_; }
  inline Price GetLowPrice() const { return low_; }
  inline Price GetClosePrice() const { return close_; }
  inline Price GetAveragePrice() const { return average_; }

  void SetTime(Time time) { time_ = time; }
  void SetOpenPrice(Price price) { open_ = price; }
  void SetHighPrice(Price price) { high_ = price; }
  void SetLowPrice(Price price) { low_ = price; }
  void SetClosePrice(Price price) { close_ = price; }
  void SetAveragePrice(Price price) { average_ = price; }

  Rate NextRate() const {
    Rate next_rate = *this;
    next_rate.time_ += TimeDifference::InMinute(1);
    next_rate.open_ = this->close_;
    next_rate.high_ = this->close_;
    next_rate.low_ = this->close_;
    next_rate.close_ = this->close_;
    next_rate.average_ = this->close_;
    return next_rate;
  }

 private:
  Time time_;
  Price open_;
  Price high_;
  Price low_;
  Price close_;
  Price average_;
};

class Rates {
 public:
  static const int kShortVolatilityDuration = 60;

  Rates() {}
  explicit Rates(const vector<Rate>& rates) : rates_(rates) {}

  size_t size() const { return rates_.size(); }

  const Rate& operator[](int index) const {
    assert(0 <= index && index < (int)rates_.size());
    return rates_[(size_t)index];
  }

  vector<Rate>::const_iterator begin() const {
    return rates_.begin();
  }

  vector<Rate>::const_iterator end() const {
    return rates_.end();
  }

  vector<Rate>::const_reverse_iterator rbegin() const {
    return rates_.rbegin();
  }

  void Load(const vector<string>& files, bool is_training) {
    int skipped_ticks = 0;
    for (const string& file : files) {
      FILE* fp = fopen(file.c_str(), "r");
      if (fp == nullptr) {
        fprintf(stderr, "Failed to read: %s\n", file.c_str());
        exit(1);
      }

      while (!feof(fp)) {
        Rate rate;
        if (!rate.Load(fp)) { break; }
        // TODO(imos): フラグで選択できるようにする
        if (!rate.GetTime().IsSafeTime(is_training)) {
          skipped_ticks++;
          continue;
        }
        // 60分以下の抜けを埋める．
        if (rates_.size() > 0) {
          int difference_in_minute =
              (int)round((rate.GetTime() -
                          rates_.back().GetTime()).GetMinute());
          if (1 < difference_in_minute && difference_in_minute <= 60) {
            for (int i = 1; i < difference_in_minute; i++) {
              rates_.push_back(rates_.back().NextRate());
            }
          }
        }
        rates_.push_back(move(rate));
      }

      fclose(fp);

      fprintf(stderr, "Successfully loaded: %s\n", file.c_str());
    }

    assert(rates_.size() > 0);
    fprintf(stderr, "- Open: %s\n", rates_.begin()->DebugString().c_str());
    fprintf(stderr, "- Close: %s\n", rates_.rbegin()->DebugString().c_str());
    fprintf(stderr, "- Ticks: %lu\n", rates_.size());
    fprintf(stderr, "- Skipped ticks: %d\n", skipped_ticks);
  }

  // [from, to] の範囲（両端を含む）を持つ Rates を返します
  Rates GetRates(int from, int to) const {
    assert(0 <= from && from < (int)rates_.size());
    assert(0 <= to && to < (int)rates_.size());
    return Rates(vector<Rate>(rates_.begin() + from, rates_.begin() + to + 1));
  }

  // from_year 年から to_year 年（両端を含む）を持つ Rates を返します．
  Rates GetRatesInYear(int from_year, int to_year) const {
    int from_index = (int)GetYearIndex(from_year);
    int to_index = (int)GetYearIndex(to_year + 1);
    if (from_index >= to_index) { return Rates(); }
    return GetRates(from_index, to_index - 1);
  }

  Volatility GetVolatility(int index) const {
    if (index < kShortVolatilityDuration) {
      return Volatility::Invalid();
    }

    Time current_time = rates_[(size_t)index].GetTime();
    Price current_price = rates_[(size_t)index].GetAveragePrice();
    VolatilitySum sum;
    for (int i = index - kShortVolatilityDuration; i < index; i++) {
      Time previous_time = rates_[(size_t)i].GetTime();
      Price previous_price = rates_[(size_t)i].GetAveragePrice();
      if ((current_time - previous_time).GetMinute()
          > kShortVolatilityDuration * 2) {
        return Volatility::Invalid();
      }
      double scale_factor = sqrt((current_time - previous_time).GetMinute());
      double price_difference =
          (current_price.GetLogPrice()
           - previous_price.GetLogPrice()) / scale_factor;
      assert(!IsNan(price_difference) && !IsInf(price_difference));
      Volatility volatility;
      volatility.SetValueDeprecated(price_difference * price_difference);
      sum += volatility;
    }
    assert(sum.GetCount() == kShortVolatilityDuration);
    return sum.GetAverageVolatility();
  }

 private:
  size_t GetYearIndex(int32_t year) const {
    Time start_time = Time::InYear(year);
    size_t left = 0;
    size_t right = rates_.size();
    while (left < right) {
      size_t middle = (left + right) / 2;
      if (rates_[middle].GetTime() < start_time) {
        left = middle + 1;
      } else {
        right = middle;
      }
    }
    return left;
  }

  vector<Rate> rates_;

  DISALLOW_COPY_AND_ASSIGN(Rates);
};

class DailyVolatility {
 public:
  static constexpr double kDailyVolatilityScale = 10000000.0;

  DailyVolatility() {}

  size_t size() const {
    return daily_volatility_sum_.size();
  }

  double operator[](int index) const {
    assert(0 <= index && index < (int)daily_volatility_sum_.size());
    return (daily_volatility_sum_[(size_t)index]
            - (index == 0 ? 0 : daily_volatility_sum_[(size_t)(index - 1)]))
           / kDailyVolatilityScale;
  }

  void Init(const Rates& rates, bool is_summer_time) {
    deque<Time> times;
    deque<Volatility> volatilities;
    vector<double> daily_volatility_sum(60 * 24, 0.0);
    vector<int32_t> daily_volatility_count(60 * 24, 0);
    VolatilitySum sum;
    for (int index = 0; index < (int)rates.size(); index++) {
      if (rates[index].GetTime().IsSummerTime() != is_summer_time) { continue; }

      Volatility volatility = rates.GetVolatility(index);
      if (!volatility.IsValid()) { continue; }
      times.push_back(rates[index].GetTime());
      volatilities.push_back(volatility);
      sum += volatility;

      while ((times.back() - times.front()).GetSecond() > 60 * 60 * 24 * 7) {
        sum -= volatilities.front();
        times.pop_front();
        volatilities.pop_front();
      }

      if (times.size() < 60 * 24 * 7 / 2) { continue; }
      int time_index = times.back().GetMinuteIndex() % (60 * 24);
      assert(0 <= time_index);
      double current_volatility = volatilities.back().GetValue();
      assert(current_volatility >= 0);
      assert(!IsNan(current_volatility) && !IsInf(current_volatility));
      double average_volatility =
          sum.GetAverageVolatility().GetValue();
      assert(average_volatility >= 0);
      assert(!IsNan(average_volatility) && !IsInf(average_volatility));
      daily_volatility_sum[(size_t)time_index]
          += current_volatility / average_volatility;
      daily_volatility_count[(size_t)time_index]++;
    }
    vector<double> daily_volatility;
    for (int i = 0; i < (int)daily_volatility_sum.size(); i++) {
      daily_volatility.push_back(
          pow(daily_volatility_sum[(size_t)i]
              / daily_volatility_count[(size_t)i], 2));
    }
    double volatility_sum = 0.0;
    for (double value : daily_volatility) {
      volatility_sum += value * value;
    }
    daily_volatility_sum_.clear();
    int64_t accumulated_daily_volatility = 0;
    for (double value : daily_volatility) {
      double scaled_value =
          round(value / sqrt(volatility_sum / daily_volatility.size())
                      * kDailyVolatilityScale);
      assert(numeric_limits<int32_t>::min() <= scaled_value &&
             scaled_value <= numeric_limits<int32_t>::max());
      accumulated_daily_volatility += (int32_t)scaled_value;
      daily_volatility_sum_.push_back(accumulated_daily_volatility);
    }
  }

  VolatilityRatio GetVolatilityBetween(Time from, Time to) const {
    int from_index = from.GetMinuteIndex() - 1;
    int to_index = to.GetMinuteIndex();
    int64_t result =
        daily_volatility_sum_[to_index % (60 * 24)]
        - daily_volatility_sum_[from_index % (60 * 24)]
        + daily_volatility_sum_[60 * 24 - 1]
            * (to_index / (60 * 24) - from_index / (60 * 24));
    return VolatilityRatio(
        result / kDailyVolatilityScale / (to_index - from_index));
  }

  string DebugString() const {
    string result;

    result += "{summary: [";
    char buf[100];
    for (int i = 0; i < (int)size(); i += 60) {
      sprintf(buf, "%.3f", (*this)[i]);
      if (i != 0) { result += ", "; }
      result += buf;
    }
    result += "], ";

    // ボラティリティの2乗の平均は 1 日を通して 1.0 となるように調整されているか確認する．
    // NOTE: ここで表示される値は 1 日で発生する誤差（0.01% 以下であれば十分）
    double sum = 0.0;
    for (int i = 0; i < (int)size(); i++) {
      sum += pow((*this)[i], 2.0);
    }
    result += "error_for_one_day: ";
    sprintf(buf, "%e", sum / size() - 1);
    result += buf;
    result += "}";

    return result;
  }

 private:
  vector<int64_t> daily_volatility_sum_;

  DISALLOW_COPY_AND_ASSIGN(DailyVolatility);
};

class AccumulatedRates {
 public:
  AccumulatedRates() : name_("unknown data"), initialized_(false) {}
  AccumulatedRates(const char* name, const Rates& rates) : name_(name) {
    Init(rates);
  }

  void Init(const Rates& rates) {
    fprintf(stderr, "Initializing AccumulatedRates for %s...\n", name_.c_str());

    assert(rates.size() > 0);
    count_.clear();
    open_.clear();
    close_.clear();
    start_time_ = rates.begin()->GetTime();
    end_time_ = rates.rbegin()->GetTime();
    vector<TimeSum> time_sum;
    vector<Price> high_data;
    vector<Price> low_data;
    vector<PriceSum> sum_data;
    vector<VolatilitySum> volatility_sum;
    for (int rate_index = 0; rate_index < (int)rates.size(); rate_index++) {
      const Rate& rate = rates[rate_index];
      int index = rate.GetTime().GetIndex(start_time_);
      assert(0 <= index);
      assert((int)count_.size() <= index);
      count_.resize((size_t)(index + 1), 0);
      count_[(size_t)index] = 1;
      time_sum.push_back(TimeSum::From(rate.GetTime()));
      high_data.push_back(rate.GetHighPrice());
      low_data.push_back(rate.GetLowPrice());
      sum_data.push_back(PriceSum::From(rate.GetAveragePrice()));
      open_.push_back(rate.GetOpenPrice());
      close_.push_back(rate.GetClosePrice());
      volatility_sum.push_back(VolatilitySum(rates.GetVolatility(rate_index)));
    }
    int32_t sum = -1;
    for (int32_t& value : count_) {
      sum = value = sum + value;
    }
    assert(count_[0] == 0);
    vector<thread> threads;
    threads.push_back(thread([&]{ time_sum_.Init(time_sum); }));
    threads.push_back(thread([&]{ high_.Init(high_data); }));
    threads.push_back(thread([&]{ low_.Init(low_data); }));
    threads.push_back(thread([&]{ sum_.Init(sum_data); }));
    threads.push_back(thread([&]{ volatility_sum_.Init(volatility_sum); }));
    for (thread& t : threads) { t.join(); }
    initialized_ = true;

    fprintf(stderr, "Initialized AccumulatedRates for %s.\n", name_.c_str());
    fprintf(stderr, "%s\n", DebugString().c_str());
  }

  void InitDailyVolatility(const Rates& rates) {
    CHECK(&rates != nullptr);
    if (GetParams().daily_volatility) {
      winter_daily_volatility_.Init(rates, false /* is_summer_time */);
      summer_daily_volatility_.Init(rates, true /* is_summer_time */);
    }
  }

  int Count(Time from, Time to) const {
    if (to < from) { return Count(to, from); }

    int from_index = GetDenseIndexOrNext(from);
    int to_index = GetDenseIndexOrPrevious(to);
    if (from_index > to_index) { return 0; }
    return to_index - from_index + 1;
  }

  Rate GetRate(Time from, Time to) const {
    if (to < from) { return GetRate(to, from); }

    int from_index = GetDenseIndexOrNext(from);
    int to_index = GetDenseIndexOrPrevious(to);
    if (from_index > to_index) { return Rate(); }

    Rate rate;
    assert(0 <= from_index && from_index < (int)count_.size());
    assert(0 <= to_index && to_index < (int)count_.size());
    assert(from_index <= to_index);
    rate.SetTime(time_sum_.Query(from_index, to_index)
                          .GetAverage(to_index - from_index + 1));
    rate.SetOpenPrice(open_[(size_t)from_index]);
    rate.SetClosePrice(close_[(size_t)to_index]);
    rate.SetHighPrice(high_.Query(from_index, to_index));
    rate.SetLowPrice(low_.Query(from_index, to_index));
    rate.SetAveragePrice(
        sum_.Query(from_index, to_index)
            .GetAverage(to_index - from_index + 1));
    CHECK(rate.IsComplete()) << "Incomplete rate: " << rate.DebugString();
    return move(rate);
  }

  Volatility GetVolatility(Time to) const {
    if (GetParams().base_volatility_interval == 0) {
      return Volatility::InRatio(1.0002);
    }
    Time from = to - TimeDifference::InMinute(
        GetParams().base_volatility_interval);

    int from_index = GetDenseIndexOrNext(from);
    int to_index = GetDenseIndexOrPrevious(to);
    if (from_index > to_index) { return Volatility::Invalid(); }
    if ((to_index - from_index)
        < GetParams().base_volatility_interval * 0.75) {
      return Volatility::Invalid();
    }

    assert(0 <= from_index && from_index < (int)count_.size());
    assert(0 <= to_index && to_index < (int)count_.size());
    assert(from_index <= to_index);
    Volatility volatility =
        volatility_sum_.Query(from_index, to_index).GetAverageVolatility();
    assert(volatility.GetRawValue() > 0);
    if (GetParams().daily_volatility) {
      volatility =
          volatility
          * (to.IsSummerTime() ? summer_daily_volatility_
                               : winter_daily_volatility_)
                .GetVolatilityBetween(
                    to - TimeDifference::InMinute(4 * 60), to);
    }
    return volatility;
  }

  string DebugString() {
    return "AccumulatedRates for " + name_ + ": " +
           GetRate(start_time_, end_time_).DebugString() + ", " +
           "volatility: " +
           volatility_sum_.Query(GetDenseIndexOrNext(GetStartTime()),
                                 GetDenseIndexOrPrevious(GetEndTime()))
                          .GetAverageVolatility()
                          .DebugString();
  }

  static void Test() {
    fprintf(stderr, "Testing AccumulatedRates...\n");

    vector<Rate> raw_rates(3);
    raw_rates[0].SetTime(Time::InMinute(1001));
    raw_rates[0].SetOpenPrice(Price::InRawValue(100));
    raw_rates[0].SetHighPrice(Price::InRawValue(120));
    raw_rates[0].SetLowPrice(Price::InRawValue(90));
    raw_rates[0].SetClosePrice(Price::InRawValue(110));
    raw_rates[0].SetAveragePrice(Price::InRawValue(105));
    raw_rates[1].SetTime(Time::InMinute(1002));
    raw_rates[1].SetOpenPrice(Price::InRawValue(105));
    raw_rates[1].SetHighPrice(Price::InRawValue(125));
    raw_rates[1].SetLowPrice(Price::InRawValue(95));
    raw_rates[1].SetClosePrice(Price::InRawValue(115));
    raw_rates[1].SetAveragePrice(Price::InRawValue(110));
    raw_rates[2].SetTime(Time::InMinute(1004));
    raw_rates[2].SetOpenPrice(Price::InRawValue(110));
    raw_rates[2].SetHighPrice(Price::InRawValue(130));
    raw_rates[2].SetLowPrice(Price::InRawValue(100));
    raw_rates[2].SetClosePrice(Price::InRawValue(120));
    raw_rates[2].SetAveragePrice(Price::InRawValue(115));

    AccumulatedRates rates{"test case", Rates(raw_rates)};
    {
      assert(rates.GetDenseIndexOrNext(Time::InMinute(1000)) == 0);
      assert(rates.GetDenseIndexOrNext(Time::InMinute(1001)) == 0);
      assert(rates.GetDenseIndexOrNext(Time::InMinute(1002)) == 1);
      assert(rates.GetDenseIndexOrNext(Time::InMinute(1003)) == 2);
      assert(rates.GetDenseIndexOrNext(Time::InMinute(1004)) == 2);
      assert(rates.GetDenseIndexOrNext(Time::InMinute(1005)) == 3);
      assert(rates.GetDenseIndexOrPrevious(Time::InMinute(1000)) == -1);
      assert(rates.GetDenseIndexOrPrevious(Time::InMinute(1001)) == 0);
      assert(rates.GetDenseIndexOrPrevious(Time::InMinute(1002)) == 1);
      assert(rates.GetDenseIndexOrPrevious(Time::InMinute(1003)) == 1);
      assert(rates.GetDenseIndexOrPrevious(Time::InMinute(1004)) == 2);
      assert(rates.GetDenseIndexOrPrevious(Time::InMinute(1005)) == 2);
    }
    {
      Rate rate = rates.GetRate(Time::InMinute(500), Time::InMinute(1000));
      assert(rates.Count(Time::InMinute(500), Time::InMinute(1000)) == 0);
      assert(!rate.IsValid());
      assert(!rate.GetTime().IsValid());
      assert(!rate.GetOpenPrice().IsValid()); 
      assert(!rate.GetHighPrice().IsValid()); 
      assert(!rate.GetLowPrice().IsValid()); 
      assert(!rate.GetClosePrice().IsValid()); 
      assert(!rate.GetAveragePrice().IsValid()); 
    }
    {
      Rate rate = rates.GetRate(Time::InMinute(1001), Time::InMinute(1004));
      assert(rates.Count(Time::InMinute(1001), Time::InMinute(1004)) == 3);
      assert(rate.IsValid());
      assert(fabs(rate.GetTime().GetMinute() - 1002.33333) < 1e-5);
      assert(rate.GetOpenPrice().GetLogPrice() == 100); 
      assert(rate.GetHighPrice().GetLogPrice() == 130); 
      assert(rate.GetLowPrice().GetLogPrice() == 90); 
      assert(rate.GetClosePrice().GetLogPrice() == 120); 
      assert(rate.GetAveragePrice().GetLogPrice() == 110); 
    }
    {
      Rate rate = rates.GetRate(Time::InMinute(1001), Time::InMinute(1002));
      assert(rates.Count(Time::InMinute(1001), Time::InMinute(1002)) == 2);
      assert(rate.IsValid());
      assert(fabs(rate.GetTime().GetMinute() - 1001.5) < 1e-3);
      CHECK_EQ(rate.GetOpenPrice().GetLogPrice(), 100);
      CHECK_EQ(rate.GetHighPrice().GetLogPrice(), 125); 
      CHECK_EQ(rate.GetLowPrice().GetLogPrice(), 90); 
      CHECK_EQ(rate.GetClosePrice().GetLogPrice(), 115); 
      CHECK_EQ(rate.GetAveragePrice().GetLogPrice(), 108); 
    }
    {
      Rate rate = rates.GetRate(Time::InMinute(1005), Time::InMinute(1005));
      assert(rates.Count(Time::InMinute(1005), Time::InMinute(1005)) == 0);
      assert(!rate.IsValid());
      assert(!rate.GetTime().IsValid());
      assert(!rate.GetOpenPrice().IsValid()); 
      assert(!rate.GetHighPrice().IsValid()); 
      assert(!rate.GetLowPrice().IsValid()); 
      assert(!rate.GetClosePrice().IsValid()); 
      assert(!rate.GetAveragePrice().IsValid()); 
    }

    fprintf(stderr, "AccumulatedRates test successfully passed.\n");
  }

  Time GetStartTime() const { return start_time_; }
  Time GetEndTime() const { return end_time_; }

  const string& GetName() const { return name_; }

 private:
  bool HasDenseIndex(int index) const {
    assert(0 <= index && index < (int)count_.size());
    return count_[(size_t)index] !=
           (index == 0 ? -1 : count_[(size_t)index - 1]);
  }

  int GetDenseIndexOrNext(Time time) const {
    if (time < start_time_) { return 0; }
    if (end_time_ < time) {
      CHECK_LT(0, count_.size());
      return (int)count_.back() + 1;
    }

    int index = time.GetIndex(start_time_);
    assert(0 <= index && index < (int)count_.size());
    return count_[(size_t)index] + (HasDenseIndex(index) ? 0 : 1);
  }

  int GetDenseIndexOrPrevious(Time time) const {
    if (time < start_time_) { return -1; }
    if (end_time_ < time) {
      CHECK_LT(0, count_.size());
      return (int)count_.back();
    }

    int index = time.GetIndex(start_time_);
    assert(0 <= index && index < (int)count_.size());
    return count_[(size_t)index];
  }

  string name_;
  DailyVolatility winter_daily_volatility_;
  DailyVolatility summer_daily_volatility_;
  SegmentTree<TimeSum, TimeSum, TimeSum::Add> time_sum_;
  SegmentTree<Price, const Price&, max<Price>> high_;
  SegmentTree<Price, const Price&, min<Price>> low_;
  SegmentTree<PriceSum, PriceSum, PriceSum::Add> sum_;
  SegmentTree<VolatilitySum, VolatilitySum, VolatilitySum::Add> volatility_sum_;
  vector<Price> open_;
  vector<Price> close_;
  vector<int32_t> count_;
  Time start_time_;
  Time end_time_;
  bool initialized_;

  DISALLOW_COPY_AND_ASSIGN(AccumulatedRates);
};

struct AdjustedPriceRoot {
  typedef int32_t ScalarType;

  struct DifferenceType : public DifferenceBase<AdjustedPriceRoot> {};

  struct ValueType : public ValueBase<AdjustedPriceRoot> {
    // TODO(imos): 削除する．
    static constexpr double kBaseAdjustedPrice = 100000000;
    static constexpr double kAdjustedPriceScale = 10000;

    ValueType() : ValueBase<AdjustedPriceRoot>() {}

    void Init(Price price,
              TimeDifference interval,
              Price base_price,
              Volatility volatility,
              double ratio = 1.0) {
      try {
        SetRawValue(
            (price - base_price).GetLogValue() * ratio
                / sqrt(fabs(interval.GetMinute())) / volatility.GetValue()
                * kAdjustedPriceScale);
      } catch (TException) {
        LOG(ERROR) <<
            "price: " << price << ", " <<
            "interval: " << interval << ", " <<
            "base_price: " << base_price << ", " <<
            "volatility: " << volatility << ", " <<
            "ratio: " << ratio;
        throw TException();
      }
    }

    bool InitFuture(const AccumulatedRates& rates,
                    Time now,
                    Time to,
                    PriceDifference price_adjustment) {
      Price current_price =
          rates.GetRate(now, now).GetClosePrice() + price_adjustment;
      if (!current_price.IsValid()) { return false; }

      Volatility current_volatility = rates.GetVolatility(now);
      if (!current_volatility.IsValid()) { return false; }

      if (GetParams().future == Params::Future::CLOSE) {
        Init(rates.GetRate(now, to).GetClosePrice(), to - now, current_price,
             current_volatility);
        return true;
      }

      Price upper_bound =
          current_volatility.GetPrice(current_price, to - now, 1);
      Price lower_bound =
          current_volatility.GetPrice(current_price, to - now, -1);

      Price final_price;
      {
        Rate rate = rates.GetRate(now + TimeDifference::InMinute(1), to);
        if (!rate.IsComplete()) {
          return false;
        }
        final_price = rate.GetClosePrice();
        if (lower_bound < rate.GetLowPrice() &&
            rate.GetHighPrice() < upper_bound) {
          Init(final_price, to - now, current_price, current_volatility);
          return true;
        }
      }

      TimeDifference max_difference = to - now;
      TimeDifference min_difference = TimeDifference::InMinute(1);
      while ((max_difference - min_difference).GetMinute() > 0.5) {
        TimeDifference median_difference = TimeDifference::InMinute(
            (max_difference.GetMinute() + min_difference.GetMinute()) / 2);
        Rate rate = rates.GetRate(now + TimeDifference::InMinute(1),
                                  now + median_difference);
        if (!rate.IsComplete()) {
          break;
        }

        final_price = rate.GetClosePrice();
        if (lower_bound < rate.GetLowPrice() &&
            rate.GetHighPrice() < upper_bound) {
          min_difference = median_difference;
        } else {
          max_difference = median_difference;
        }
      }

      switch (GetParams().future) {
        case Params::Future::CROSS: {
          Rate rate = rates.GetRate(now + TimeDifference::InMinute(1),
                                    now + max_difference);
          final_price = rate.GetClosePrice();
          break;
        }
        case Params::Future::LIMIT: {
          Rate rate = rates.GetRate(now + TimeDifference::InMinute(1),
                                    now + max_difference);
          // 1分以下の変動は対応ができないので，しきい値を設けない
          // NOTE: 効果が不明のためコメントアウト
          // if (max_difference < TimeDifference::InMinute(1.9)) {
          //   final_price = rate.GetClosePrice();
          //   break;
          // }
          if (rate.GetLowPrice() < lower_bound &&
              upper_bound < rate.GetHighPrice()) {
            final_price = rate.GetClosePrice();
            if (rate.GetClosePrice() < lower_bound) {
              final_price = lower_bound;
            } else if (upper_bound < rate.GetClosePrice()) {
              final_price = upper_bound;
            }
          } else if (rate.GetLowPrice() < lower_bound) {
            final_price = lower_bound;
          } else if (upper_bound < rate.GetLowPrice()) {
            final_price = upper_bound;
          } else {
            final_price = rate.GetClosePrice();
          }
          break;
        }
        case Params::Future::CLOSE: {
          LOG(FATAL) << "This should not happen.";
        }
      }

      switch (GetParams().future_curve) {
        case Params::FutureCurve::FLAT: {
          Init(final_price, to - now, current_price, current_volatility);
          break;
        }
        case Params::FutureCurve::SQRT: {
          Init(final_price, max_difference, current_price, current_volatility);
          break;
        }
        case Params::FutureCurve::LINEAR: {
          Init(final_price, to - now, current_price, current_volatility,
               ((to - now).GetMinute() + 1) / (max_difference.GetMinute() + 1));
          break;
        }
        case Params::FutureCurve::LINEAR_CUT: {
          Init(final_price, to - now, current_price, current_volatility,
               ((to - now).GetMinute() - max_difference.GetMinute() + 1) /
                   (max_difference.GetMinute() + 1));
          break;
        }
      }
      return true;
    }

    Price GetPrice(TimeDifference interval,
                   Price base_price,
                   Volatility volatility) const {
      PriceDifference price_difference;
      price_difference.SetLogValue(GetRawValue() / kAdjustedPriceScale);
      price_difference *= sqrt(fabs(interval.GetMinute()));
      price_difference *= volatility.GetValue();
      return base_price + price_difference;
    }

    // 表示用
    Price GetRegularizedPrice(
        TimeDifference interval = TimeDifference::InMinute(1)) const {
      return GetPrice(
          interval, Price::InRealPrice(1.0), Volatility::InRatio(1e-4));
    }

    double GetRatio() const {
      return GetRegularizedPrice().GetRealPrice();
    }

    double MeasureDistance(ValueType value) const {
      double price_difference =
          (GetRawValue() - value.GetRawValue()) / kBaseAdjustedPrice;
      return price_difference * price_difference;
    }

    string DebugString() const {
      return GetRegularizedPrice(TimeDifference::InMinute(60)).DebugString();
    }
  };

  struct SumType : public SumBase<AdjustedPriceRoot> {
    SumType() : SumBase<AdjustedPriceRoot>() {}
  };
};

typedef typename AdjustedPriceRoot::DifferenceType AdjustedPriceDifference;
typedef typename AdjustedPriceRoot::ValueType AdjustedPrice;
typedef typename AdjustedPriceRoot::SumType AdjustedPriceSum;

TEST(AdjustedPrice) {
  // 価格解像度を確認（）
  {
    AdjustedPrice price;
    price.SetRawValue(1);
    double ratio = price.GetRatio();
    fprintf(stderr, "- Minimal resolution: %.3e\n", ratio - 1);
    fprintf(stderr, "- Maximal resolution: %.3e\n",
            (ratio - 1) * numeric_limits<int32_t>::max());
    CHECK_LE(1, ratio);
    // 0.1pipsの差が表現可能であるか
    CHECK_LE(ratio, 1.00001) << "AdjustedPrice cannot represents 0.1 pips "
                             << "difference in one minute.";
    // exp(10) が表現可能であるかどうか
    CHECK_GT((ratio - 1) * numeric_limits<int32_t>::max(), 10);
  }

  // 価格の復元
  {
    AdjustedPrice price;
    price.Init(Price::InRealPrice(100.01),
               TimeDifference::InMinute(1.0),
               Price::InRealPrice(100),
               Volatility::InRatio(log(100.01 / 100)));
    CHECK_NEAR(
        price.GetPrice(
            TimeDifference::InMinute(1.0),
            Price::InRealPrice(100),
            Volatility::InRatio(log(100.01 / 100))).GetRealPrice(),
        100.01,
        1e-4);
    CHECK_NEAR(
        price.GetPrice(
            TimeDifference::InMinute(1.0),
            Price::InRealPrice(200),
            Volatility::InRatio(log(100.01 / 100))).GetRealPrice(),
        200.02,
        1e-4);
    CHECK_NEAR(
        price.GetPrice(
            TimeDifference::InMinute(4.0),
            Price::InRealPrice(100),
            Volatility::InRatio(log(100.01 / 100))).GetRealPrice(),
        100.02,
        1e-4);
    CHECK_NEAR(
        price.GetPrice(
            TimeDifference::InMinute(1.0),
            Price::InRealPrice(100),
            Volatility::InRatio(log(100.02 / 100))).GetRealPrice(),
        100.02,
        1e-4);
  }
}

struct AdjustedPriceStat {
 public:
  AdjustedPriceStat()
      : adjusted_price_count_(0),
        adjusted_price_sum_(0),
        adjusted_price_sum2_(0) {}

  void Add(AdjustedPrice price) {
    assert(price.IsValid());
    adjusted_price_count_++;
    int64_t value = price.GetRawValue();
    adjusted_price_sum_ += value;
    adjusted_price_sum2_ += value * value;
  }

  string DebugString() {
    char buf[100];
    double average =
        static_cast<double>(adjusted_price_sum_) / adjusted_price_count_;
    double variance =
        static_cast<double>(adjusted_price_sum2_) / adjusted_price_count_ -
        average * average;
    sprintf(buf, "%+.3f±%.3f (%d)",
            average, sqrt(variance), adjusted_price_count_);
    return buf;
  }

  AdjustedPrice GetLowerBound() const { return GetBiasedPrice(-0.0); }
  AdjustedPrice GetUpperBound() const { return GetBiasedPrice(+0.0); }

 private:
  AdjustedPrice GetBiasedPrice(double bias) const {
    double average =
        static_cast<double>(adjusted_price_sum_) / adjusted_price_count_;
    double variance =
        static_cast<double>(adjusted_price_sum2_) / adjusted_price_count_ -
        average * average;
    AdjustedPrice result;
    result.SetRawValue(static_cast<int64_t>(
        round(average + sqrt(variance) * bias)));
    return result;
  }

  int32_t adjusted_price_count_;
  int64_t adjusted_price_sum_;
  int64_t adjusted_price_sum2_;
};

struct AdjustedRate {
 public:
  bool Init(const AccumulatedRates& rates,
            Time from,
            Time to,
            Time now,
            PriceDifference price_adjustment = PriceDifference::InRatio(1)) {
    Price current_price = rates.GetRate(now, now).GetClosePrice();
    if (!current_price.IsValid()) { return false; }
    current_price += price_adjustment;

    Volatility current_volatility = rates.GetVolatility(now);
    if (!current_volatility.IsValid()) { return false; }

    if (rates.Count(from, to) < (fabs((to - from).GetMinute()) + 1) * 0.7) {
      return false;
    }
    Rate rate = rates.GetRate(from, to);
    CHECK(rate.IsValid() && rate.IsComplete())
        << "Invalid rate: from=" << from.DebugString() << ", "
        << "to=" << to.DebugString() << ", "
        << "rate=" << rate.DebugString();
    high_.Init(rate.GetHighPrice(), now - rate.GetTime(),
               current_price, current_volatility);
    low_.Init(rate.GetLowPrice(), now - rate.GetTime(),
              current_price, current_volatility);
    average_.Init(rate.GetAveragePrice(), now - rate.GetTime(),
                  current_price, current_volatility);
    return true;
  }

  double MeasureDistance(const AdjustedRate& rate) const {
    double average_distance =
        GetAveragePrice().MeasureDistance(rate.GetAveragePrice());
    if (kHighAndLowDistanceWeight > 0.0) {
      double high_and_low_distance =
          GetHighPrice().MeasureDistance(rate.GetHighPrice()) +
          GetLowPrice().MeasureDistance(rate.GetLowPrice());
      return (1 - kHighAndLowDistanceWeight) * average_distance +
             kHighAndLowDistanceWeight * high_and_low_distance;
    }
    return average_distance;
  }

  bool IsValid() const {
    return high_.IsValid() && low_.IsValid() && average_.IsValid();
  }

  AdjustedPrice GetHighPrice() const { return high_; }
  AdjustedPrice GetLowPrice() const { return low_; }
  AdjustedPrice GetAveragePrice() const { return average_; }

  void SetHighPrice(AdjustedPrice price) { high_ = price; }
  void SetLowPrice(AdjustedPrice price) { low_ = price; }
  void SetAveragePrice(AdjustedPrice price) { average_ = price; }

  string DebugString() const {
    return "{high: " + high_.DebugString() + ", " +
           "low: " + low_.DebugString() + ", " +
           "avg: " + average_.DebugString() + "}";
  }

 private:
  AdjustedPrice high_;
  AdjustedPrice low_;
  AdjustedPrice average_;
};

struct AdjustedRateSum {
 public:
  AdjustedRateSum() {}
  AdjustedRateSum(const AdjustedRate& t)
      : high_(AdjustedPriceSum::From(t.GetHighPrice())),
        low_(AdjustedPriceSum::From(t.GetLowPrice())),
        average_(AdjustedPriceSum::From(t.GetAveragePrice())) {}

  AdjustedRate GetAverage(int count) const {
    AdjustedRate result;
    result.SetHighPrice(high_.GetAverage(count));
    result.SetLowPrice(low_.GetAverage(count));
    result.SetAveragePrice(average_.GetAverage(count));
    return result;
  }

  AdjustedRateSum operator+(const AdjustedRateSum& x) const {
    return AdjustedRateSum(high_ + x.high_,
                           low_ + x.low_,
                           average_ + x.average_);
  }

  // SegmentTree用の加算関数．
  // NOTE: SegmentTreeの関数はconst参照渡しのみ対応．
  static AdjustedRateSum Add(const AdjustedRateSum& a,
                             const AdjustedRateSum& b) {
    return a + b;
  }

 private:
  AdjustedRateSum(const AdjustedPriceSum& high,
                  const AdjustedPriceSum& low,
                  const AdjustedPriceSum& average)
      : high_(high), low_(low), average_(average) {}

  AdjustedPriceSum high_;
  AdjustedPriceSum low_;
  AdjustedPriceSum average_;
};

struct FeatureConfig {
 public:
  static constexpr size_t kFeatureSize = 3;

  FeatureConfig() : FeatureConfig({}, 0) {}

  FeatureConfig(const vector<int>& past, int future) {
    CHECK_LE(past.size(), kFeatureSize);
    for (size_t i = 0; i < kFeatureSize; i++) {
      past_[i] = i < past.size() ? past[i] : 0;
    }
    past_size_ = past.size();
    future_ = future;
  }

  TimeDifference GetPast(int index, int ratio) const {
    CHECK_LE(0, index);
    CHECK_LT(index, past_size_);
    return TimeDifference::InMinute(past_[index] * ratio);
  }

  TimeDifference GetFuture(int ratio) const {
    return TimeDifference::InMinute(future_ * ratio);
  }

  size_t GetPastSize() const { return past_size_; }

 private:
  int past_[kFeatureSize];
  size_t past_size_;
  int future_;
};

struct PastFeature {
 public:
  static constexpr size_t kFeatureSize = FeatureConfig::kFeatureSize;

  bool Init(const FeatureConfig& config,
            const AccumulatedRates& rates,
            Time now,
            Volatility volatility,
            int ratio,
            PriceDifference price_adjustment = PriceDifference::InRatio(1)) {
    if (!volatility.IsValid()) { return false; }

    TimeDifference last_difference = TimeDifference::InMinute(0);
    for (size_t i = 0; i < config.GetPastSize(); i++) {
      TimeDifference past_differnce = config.GetPast((int)i, ratio);
      if (GetParams().volatility_adjustment) {
        past_differnce /= pow(volatility.GetValue() / GetParams().spread, 2.0);
      }
      if (!past_[i].Init(rates,
                         now - past_differnce,
                         now - last_difference - TimeDifference::InMinute(1),
                         now,
                         price_adjustment)) {
        return false;
      }
      CHECK(past_[i].IsValid())
          << "past_[" << i << "] is invalid: "
          << "now=" << now.DebugString() << ", "
          << "price_adjustment=" << price_adjustment.GetRawValue();
      last_difference = past_differnce;
    }
    return true;
  }

  double MeasureDistance(const FeatureConfig& config,
                         const PastFeature& feature) const {
    double distance = 0.0;
    for (size_t i = 0; i < config.GetPastSize(); i++) {
      DCHECK(past_[i].IsValid());
      DCHECK(feature.past_[i].IsValid());
      distance += past_[i].MeasureDistance(feature.past_[i]);
    }
    return distance;
  }

  string DebugString(const string& indent = "") const {
    string result = "[\n";
    for (size_t i = 0; i < kFeatureSize; i++) {
      result += indent + "  " + past_[i].DebugString() + ",\n";
    }
    result += indent + "]";
    return result;
  }

  const AdjustedRate& GetPastRate(int index) const {
    CHECK_LE(0, index);
    CHECK_LT(index, kFeatureSize);
    return past_[index];
  }

  AdjustedRate* MutablePastRate(int index) {
    CHECK_LE(0, index);
    CHECK_LT(index, kFeatureSize);
    return &past_[index];
  }

 private:
  AdjustedRate past_[kFeatureSize];
};

struct QFeatureScore {
 public:
  QFeatureScore()
      : score_(0),
        volatility_score_(0) {
    Begin();
  }

  void Begin() {
    xy_sum_ = 0;
    xx_sum_ = 0;
    x_sum_ = 0;
    y_sum_ = 0;
    sum_ = 0;
  }

  PriceDifference GetScore(Volatility volatility) const {
    PriceDifference result;
    double score =
        score_ + volatility_score_ * pow(volatility.GetValue(),
                                         GetParams().volatility_power);
    try {
      result.SetLogValue(score);
    } catch (TException) {
      LOG(ERROR) << "score: " << score << ", "
                 << "volatility: " << volatility;
      throw TException();
    }
    return result;
  }

  void Commit(double learning_rate = 0.1) {
    if (sum_ < 10) {
      LOG(ERROR) << "Bad commit: " << sum_;
      Begin();
      return;
    }

    double covariance = xy_sum_ / sum_ - x_sum_ / sum_ * y_sum_ / sum_;
    double variance = xx_sum_ / sum_ - x_sum_ / sum_ * x_sum_ / sum_;
    CHECK(!IsNan(covariance) && !IsInf(covariance))
        << "Invalid covariance: " << covariance << ", "
        << DebugString();

    double volatility_score = covariance / variance;
    double score = y_sum_ / sum_ - x_sum_ / sum_ * volatility_score;
    CHECK(!IsNan(volatility_score) && !IsInf(volatility_score))
        << "Invalid volatility_score: " << volatility_score
        << ", " << DebugString();
    CHECK(!IsNan(score) && !IsInf(score))
        << "Invalid score: " << score << ", " << DebugString();

    score_ = score_ * (1 - learning_rate) + score * learning_rate;
    volatility_score_ = volatility_score_ * (1 - learning_rate) +
                        volatility_score * learning_rate;
    CHECK(!IsNan(score_) && !IsInf(score_)) << DebugString();
    CHECK(!IsNan(volatility_score_) && !IsInf(volatility_score_))
        << DebugString();

    Begin();
  }

  void Record(Volatility volatility, PriceDifference reward) {
    CHECK(volatility.IsValid())
        << "Invalid volatility: " << volatility.DebugString();
    CHECK_LT(PriceDifference::InRatio(0.66), reward);
    CHECK_LT(reward, PriceDifference::InRatio(1.5));

    double x = pow(volatility.GetValue(), GetParams().volatility_power);
    double y = reward.GetLogValue();
    xy_sum_ += x * y;
    xx_sum_ += x * x;
    x_sum_ += x;
    y_sum_ += y;
    sum_ += 1;
  }

  int GetWeight() const { return round(sum_); }

  string DebugString() const {
    return "{\"score\": " + StringPrintf("%g", score_) + ", " +
           "\"volatility_score\": " +
           StringPrintf("%g", volatility_score_) + ", " +
           "\"xy_sum\": " + StringPrintf("%g", xy_sum_) + ", " +
           "\"xx_sum\": " + StringPrintf("%g", xx_sum_) + ", " +
           "\"x_sum\": " + StringPrintf("%g", x_sum_) + ", " +
           "\"y_sum\": " + StringPrintf("%g", y_sum_) + ", " +
           "\"sum\": " + StringPrintf("%g", sum_) + "}";
  }

 private:
  double score_;
  double volatility_score_;

  double xy_sum_;
  double xx_sum_;
  double x_sum_;
  double y_sum_;
  double sum_;
};

struct QFeature {
 public:
  QFeature() {}

  bool Init(const FeatureConfig& config,
            const AccumulatedRates& rates,
            Time now,
            Volatility volatility,
            int ratio) {
    return past_.Init(config, rates, now, volatility, ratio);
  }

  double MeasureDistance(const FeatureConfig& config,
                         const QFeature& feature) const {
    return past_.MeasureDistance(config, feature.past_);
  }

  const QFeatureScore& GetQFeatureScore(int index) const {
    CHECK_LE(-1, index);
    CHECK_LE(index, 1);
    return score_[index + 1];
  }

  QFeatureScore* MutableQFeatureScore(int index) {
    CHECK_LE(-1, index);
    CHECK_LE(index, 1);
    return &score_[index + 1];
  }

  string DebugString() const {
    string result = "{\n  \"past\": " + past_.DebugString() + ",\n  ";
    result += "score: [\n";
    for (int i = 0; i < 3; i++) {
      result += "    " + score_[i].DebugString() + ",\n";
    }
    result += "  ]\n,}";
    return result;
  }

 private:
  PastFeature past_;
  QFeatureScore score_[3];
};

struct QFeatureReward {
 public:
  QFeatureReward() : next_reward_(nullptr) {
    for (auto& feature_id : feature_ids_) {
      feature_id = -1;
    }
    for (auto& score : scores_) {
      score = PriceDifference::InRatio(1);
    }
  }

  bool Init(const vector<QFeature>& features,
            const FeatureConfig& config,
            const AccumulatedRates& rates,
            Time now,
            int ratio) {
    now_ = now;
    volatility_ = rates.GetVolatility(now);
    if (!volatility_.IsValid()) {
      return false;
    }

    QFeature feature;
    if (!feature.Init(config, rates, now, volatility_, ratio)) {
      return false;
    }

    Price current_price = rates.GetRate(now, now).GetClosePrice();
    if (!current_price.IsValid()) {
      return false;
    }
    price_ = current_price;

    Time next_time = now + TimeDifference::InMinute(1);
    Price next_price = rates.GetRate(next_time, next_time).GetClosePrice();
    if (!next_price.IsValid()) {
      return false;
    }
    reward_ = (next_price - current_price) * sqrt(1.0 / ratio);

    vector<pair<double, int>> distance_and_feature_ids;
    for (int feature_id = 0; feature_id < (int)features.size(); feature_id++) {
      distance_and_feature_ids.emplace_back(
          features[feature_id].MeasureDistance(config, feature),
          feature_id);
    }
    sort(distance_and_feature_ids.begin(), distance_and_feature_ids.end());
    CHECK_LE(feature_ids_.size(), distance_and_feature_ids.size());
    for (size_t feature_id_index = 0;
         feature_id_index < feature_ids_.size(); feature_id_index++) {
      feature_ids_[feature_id_index] =
          distance_and_feature_ids[feature_id_index].second;
    }

    return true;
  }

  bool IsValid() const {
    return feature_ids_[0] >= 0;
  }

  Time GetTime() const { return now_; }
  Price GetPrice() const { return price_; }

  // TODO(imos): Deprecate this.  Use GetFeatureIds instead.
  // int GetFeatureId() const {
  //   return feature_ids_[0];
  // }
  const array<int, kQRedundancy>& GetFeatureIds() const { return feature_ids_; }

  PriceDifference GetReward(int ratio) const { return reward_ * sqrt(ratio); }
  Volatility GetVolatility() const { return volatility_; }

  void SetNextReward(const QFeatureReward* next_reward)
      { next_reward_ = CHECK_NOTNULL(next_reward); }
  bool HasNextReward() const { return next_reward_ != nullptr; }
  const QFeatureReward& GetNextReward() const
      { return *CHECK_NOTNULL(next_reward_); }

  PriceDifference GetScore(int leverage) const {
    CHECK_LE(-1, leverage);
    CHECK_LE(leverage, 1);
    return scores_[leverage + 1];
  }
  void SetScore(int leverage, PriceDifference score) {
    CHECK_LE(-1, leverage);
    CHECK_LE(leverage, 1);
    scores_[leverage + 1] = score;
  }

 private:
  Time now_;
  Price price_;
  array<int, kQRedundancy> feature_ids_;
  PriceDifference reward_;
  Volatility volatility_;
  array<PriceDifference, 3> scores_;
  const QFeatureReward* next_reward_;
};

class QFeatures {
 public:
  QFeatures() {}
  QFeatures(const vector<QFeature>& features,
            const FeatureConfig& config,
            const string& name)
      : name_(name), config_(config), features_(features) {
    fprintf(stderr, "QFeatures for %s\n", name_.c_str());
  }

  void Init(const FeatureConfig& config,
            const AccumulatedRates& rates,
            int ratio,
            int ratio_range) {
    name_ = rates.GetName();
    config_ = config;
    fprintf(stderr, "Generating Q-features for %s...\n", name_.c_str());

    set<Time> times;
    while (features_.size() < kQFeatureSize) {
      int interval = rates.GetEndTime().GetMinuteIndex() -
                     rates.GetStartTime().GetMinuteIndex();
      Time time = rates.GetStartTime() +
          TimeDifference::InMinute(
              static_cast<int32_t>(Rand() % interval));
      if (times.count(time) > 0) {
        continue;
      }
      times.insert(time);
      QFeature feature;
      if (feature.Init(config, rates, time, rates.GetVolatility(time), ratio)) {
        features_.push_back(feature);
      }
    }
    fprintf(stderr, "- # of features for %s is %lu.\n",
            rates.GetName().c_str(), features_.size());

    vector<int> ratios;
    for (int r = ratio - ratio_range; r <= ratio + ratio_range;
         r += max(1, ratio_range / 2)) {
      ratios.push_back(r);
    }
    TimeIterator iterator(
        rates.GetStartTime(), rates.GetEndTime(), true /* show_progress */);
    rewards_.resize(ratios.size(), vector<QFeatureReward>(iterator.Count()));
    Parallel([&](int){
      Time time = iterator.GetAndIncrement();
      if (!time.IsValid()) { return false; }
      for (size_t ratio_index = 0; ratio_index < ratios.size(); ratio_index++) {
        int reward_index =
            time.GetMinuteIndex() - rates.GetStartTime().GetMinuteIndex();
        CHECK_LE(0, reward_index);
        CHECK_LT(reward_index, rewards_[ratio_index].size());
        rewards_[ratio_index][reward_index]
            .Init(features_, config, rates, time, ratios[ratio_index]);
      }
      return true;
    });

    // 転置インデックス (feature_rewards_) の生成
    feature_rewards_ = vector<vector<QFeatureReward*>>(features_.size());
    for (vector<QFeatureReward>& ratio_rewards : rewards_) {
      for (size_t reward_index = 0; reward_index + 1 < ratio_rewards.size();
           reward_index++) {
        ratio_rewards[reward_index].SetNextReward(
            &ratio_rewards[reward_index + 1]);
        for (int feature_index : ratio_rewards[reward_index].GetFeatureIds()) {
          if (feature_index < 0) { continue; }
          feature_rewards_[feature_index].push_back(
              &ratio_rewards[reward_index]);
        }
      }
    }

    LOG(INFO) << "Q-features for " << name_ << " are initialized.";
  }

  size_t size() const { return features_.size(); }
  vector<QFeature>::const_iterator begin() const { return features_.begin(); }
  vector<QFeature>::const_iterator end() const { return features_.end(); }

  const FeatureConfig& GetFeatureConfig() const { return config_; }

  void Simulate(const AccumulatedRates& rates,
                int ratio) {
    // int leverage = 0;
    Asset asset;
    Price last_price;
    int last_week_index = 0;
    for (Time now = rates.GetStartTime(); now < rates.GetEndTime();
         now.AddMinute(1)) {
      Price current_price = rates.GetRate(now, now).GetClosePrice();
      if (current_price.IsValid() && last_week_index != now.GetWeekIndex()) {
        LOG(INFO) << now.DebugString() << "\t"
                  << StringPrintf("%.6f\t", asset.GetValue(current_price))
                  << current_price.DebugString();
        last_week_index = now.GetWeekIndex();
      }

      // 最初に有効な価格が設定されるまでは何もしない
      if (!last_price.IsValid()) {
        if (current_price.IsValid()) {
          last_price = current_price;
        }
        continue;
      }
      CHECK(last_price.IsValid()) << last_price;

      // 現在の特徴量を生成する
      QFeature feature;
      Volatility volatility = rates.GetVolatility(now);
      if (!volatility.IsValid() ||
          !feature.Init(config_, rates, now, volatility, ratio) ||
          !current_price.IsValid()) {
        // leverage = 0;
        asset.Trade(now, last_price, 0, true);
        continue;
      }
      last_price = current_price;
      CHECK(current_price.IsValid()) << current_price;

      vector<pair<double, int>> distance_and_feature_ids;
      for (size_t feature_id = 0; feature_id < features_.size(); feature_id++) {
        distance_and_feature_ids.emplace_back(
            features_[feature_id].MeasureDistance(config_, feature),
            feature_id);
      }
      sort(distance_and_feature_ids.begin(), distance_and_feature_ids.end());

      PriceDifference score = PriceDifference::InRatio(1);
      for (int i = 0; i < kQRedundancy; i++) {
        QFeature& predicted_feature =
            features_[distance_and_feature_ids[i].second];
        score += predicted_feature.MutableQFeatureScore(1)
                                  ->GetScore(volatility);
        score -= predicted_feature.MutableQFeatureScore(-1)
                                  ->GetScore(volatility);
      }
      double ratio = score.GetLogValue() / kQRedundancy / GetParams().spread;
      double current_leverage = asset.GetLeverage(current_price);
      double leverage = current_leverage;
      if (ratio * current_leverage < 0) { leverage = 0; }
      if (ratio > 1) { leverage = max(leverage, (ratio - 1) / 2); }
      if (ratio < -1) { leverage = min(leverage, (ratio + 1) / 2); }
      leverage = max(-1.0, min(1.0, leverage));
      asset.Trade(now, current_price, leverage, true);
      /*
      PriceDifference best_score = PriceDifference::InRatio(0.1);
      int best_leverage = 0;
      for (int leverage_to = -1; leverage_to <= 1; leverage_to++) {
        PriceDifference score =
            PriceDifference::InRatio(1 - GetParams().spread) *
            (min(1, abs(leverage_to - leverage)) *
             abs(leverage_to) * kQRedundancy);
        for (int i = 0; i < kQRedundancy; i++) {
          QFeature& predicted_feature =
              features_[distance_and_feature_ids[i].second];
          score += predicted_feature.MutableQFeatureScore(leverage_to)
                                    ->GetScore(volatility);
        }
        if (best_score < score) {
          best_score = score;
          best_leverage = leverage_to;
        }
      }

      int next_leverage = best_leverage;

      if (leverage != next_leverage) {
        VLOG(1) << now.DebugString() << ": "
                << current_price.DebugString() << ": "
                << StringPrintf("%.6f", asset.GetValue(current_price))
                << StringPrintf(": %+d => %+d", leverage, next_leverage);
      }
      leverage = next_leverage;
      asset.Trade(now, current_price, next_leverage, true);
      */
    }
    LOG(INFO) << asset.Stats();
  }

  void LearnReward(const QFeatureReward& reward, int feature_id, int ratio) {
    if (!reward.IsValid() || !reward.HasNextReward()) { return; }
    const QFeatureReward& next_reward = reward.GetNextReward();
    if (!next_reward.IsValid()) { return; }

    const double kDiscountFactor = 0.97;
    QFeature* feature = &features_[feature_id];
    for (int leverage_from = -1; leverage_from <= 1; leverage_from++) {
      PriceDifference best_score;
      for (int leverage_to = -1; leverage_to <= 1; leverage_to++) {
        PriceDifference score =
            next_reward.GetScore(leverage_to) * kDiscountFactor +
            PriceDifference::InRatio(1 - GetParams().spread) *
            (min(1, abs(leverage_to - leverage_from)) *
             abs(leverage_to) * 0 +
             abs(leverage_to - leverage_from) * 30) +
            reward.GetReward(ratio) * leverage_to;
        if (leverage_to == -1 || best_score < score) {
          best_score = score;
        }
      }
      feature->MutableQFeatureScore(leverage_from)
             ->Record(reward.GetVolatility(), best_score);
    }
  }

  void CalculateScore() {
    ParallelFor<size_t>(
      0 /* begin */, rewards_[0].size() /* end */,
      1 /* step */, 1440 /* big_step */,
      [&](size_t reward_id) {
        for (size_t ratio_id = 0; ratio_id < rewards_.size(); ratio_id++) {
          CHECK_LT(reward_id, rewards_[ratio_id].size());
          QFeatureReward* reward = &rewards_[ratio_id][reward_id];
          if (!reward->IsValid()) {
            continue;
          }
          for (int leverage = -1; leverage <= 1; leverage++) {
            PriceDifference score = PriceDifference::InRatio(1);
            for (int feature_id : reward->GetFeatureIds()) {
              CHECK_LE(0, feature_id);
              score += features_[feature_id].MutableQFeatureScore(leverage)
                                            ->GetScore(reward->GetVolatility());
            }
            reward->SetScore(leverage, score / reward->GetFeatureIds().size());
          }
        }
      });
    int hash = 0;
    for (size_t ratio_id = 0; ratio_id < rewards_.size(); ratio_id++) {
      for (const QFeatureReward& reward : rewards_[ratio_id]) {
        if (!reward.IsValid()) {
          continue;
        }
        for (int leverage = -1; leverage <= 1; leverage++) {
          PriceDifference score = reward.GetScore(leverage);
          hash ^= score.GetRawValue();
        }
      }
    }
    LOG(ERROR) << "Hash: " << hash;
  }

  void Learn(double learning_rate, int ratio) {
    CalculateScore();

    ParallelFor<size_t>(
      0 /* begin */, feature_rewards_.size() /* end */,
      1 /* step */, 1 /* big_step */,
      [&](size_t feature_id) {
        for (const QFeatureReward* reward : feature_rewards_[feature_id]) {
          LearnReward(*reward, feature_id, ratio);
        }
        for (int leverage = -1; leverage <= 1; leverage++) {
          features_[feature_id].MutableQFeatureScore(leverage)
              ->Commit(learning_rate);
        }
      });

    for (int leverage = -1; leverage <= 1; leverage++) {
      double sum = 0, count = 0;
      for (const auto& feature : features_) {
        sum += feature.GetQFeatureScore(leverage)
                      .GetScore(Volatility::InRatio(1.5e-4))
                      .GetLogValue();
        count++;
      }
      LOG(INFO) << "Score bias: " << StringPrintf("%.6f", sum / count)
                << " (leverage=" << leverage << ")";
    }
  }

 private:
  string name_;
  FeatureConfig config_;
  vector<QFeature> features_;
  vector<vector<QFeatureReward>> rewards_;

  // features_ のインデックスから rewards_ の中へのポインター．
  // NOTE: 並列化のため更新対象となる QFeature でインデックスされた QFeatureReward の
  // ポインター群を保存している．
  vector<vector<QFeatureReward*>> feature_rewards_;

  // NOTE: feature_rewards_ がポインターを持つためコピーを禁止している
  DISALLOW_COPY_AND_ASSIGN(QFeatures);
};

struct Feature {
 public:
  static constexpr size_t kFeatureSize = FeatureConfig::kFeatureSize;

  Feature() : weight_(0) {}

  bool InitFuture(const AccumulatedRates& rates,
                  Time now,
                  TimeDifference difference,
                  PriceDifference price_adjustment) {
    if (!GetParams().future_variation) {
      return future_.InitFuture(
          rates, now, now + difference, price_adjustment);
    }
    const double kRatio = pow(2.0, 0.2);
    AdjustedPriceSum future_price_sum;
    int count = 0;
    for (double ratio = 1 / 2.0; ratio < 2; ratio *= kRatio) {
      AdjustedPrice future_price;
      if (!future_price.InitFuture(
              rates, now,
              now + TimeDifference::InMinute(difference.GetMinute() * ratio),
              price_adjustment)) {
        return false;
      }
      future_price_sum += AdjustedPriceSum::From(future_price);
      count++;
    }
    future_ = future_price_sum.GetAverage(count);
    return true;
  }

  bool Init(const FeatureConfig& config,
            const AccumulatedRates& rates,
            Time now,
            int ratio,
            PriceDifference price_adjustment = PriceDifference::InRatio(1)) {
    past_.Init(
        config, rates, now, rates.GetVolatility(now), ratio, price_adjustment);
    if (!InitFuture(rates, now, config.GetFuture(ratio), price_adjustment)) {
      return false;
    }
    assert(future_.IsValid());
    weight_ = 1;
    return true;
  }

  double MeasureDistance(const FeatureConfig& config,
                         const Feature& feature) const {
    return past_.MeasureDistance(config, feature.past_);
  }

  const PastFeature& GetPastFeature() const { return past_; }
  const AdjustedPrice& GetFuturePrice() const { return future_; }
  int32_t GetWeight() const { return weight_; }

  PastFeature* MutablePastFeature() { return &past_; }

  void ClearFuturePrice() { future_.Clear(); }

  string DebugString(const string& indent) const {
    string result = "feature: {\n" + indent + "  past: ";
    result += past_.DebugString(indent + "  ") + ",\n";
    result += indent + "  future: " + future_.DebugString() + ",\n";
    result += indent + "  weight: ";

    char buf[12];
    sprintf(buf, "%d", weight_);
    result += buf;

    result += ",\n" + indent + "}\n";
    return result;
  }

 private:
  PastFeature past_;
  AdjustedPrice future_;
  int32_t weight_;

  friend struct FeatureSum;
};

struct FeatureSum {
 public:
  static constexpr size_t kFeatureSize = Feature::kFeatureSize;

  FeatureSum() {}
  FeatureSum(const Feature& t) {
    for (size_t i = 0; i < kFeatureSize; i++) {
      past_[i] = AdjustedRateSum(t.GetPastFeature().GetPastRate(i));
    }
    future_ = AdjustedPriceSum::From(t.future_);
  }

  Feature GetAverage(int count) const {
    Feature result;
    for (size_t i = 0; i < kFeatureSize; i++) {
      *result.MutablePastFeature()->MutablePastRate(i) =
          past_[i].GetAverage(count);
    }
    result.future_ = future_.GetAverage(count);
    result.weight_ = count;
    return result;
  }

  const FeatureSum& operator+=(const FeatureSum& x) {
    for (size_t i = 0; i < kFeatureSize; i++) {
      past_[i] = past_[i] + x.past_[i];
    }
    future_ = future_ + x.future_;
    return *this;
  }

 private:
  AdjustedRateSum past_[kFeatureSize];
  AdjustedPriceSum future_;
};

struct AdjustedPriceWeightedSum {
 public:
  AdjustedPriceWeightedSum() : adjusted_price_sum_(0), weight_sum_(0) {}
  AdjustedPriceWeightedSum(AdjustedPrice adjusted_price, int32_t weight)
      : adjusted_price_sum_((int64_t)adjusted_price.GetRawValue() * weight),
        weight_sum_(weight) {}

  AdjustedPriceWeightedSum operator+(const AdjustedPriceWeightedSum& x) const {
    AdjustedPriceWeightedSum result = *this;
    result += x;
    return result;
  }

  const AdjustedPriceWeightedSum& operator+=(
      const AdjustedPriceWeightedSum& x) {
    this->adjusted_price_sum_ += x.adjusted_price_sum_;
    assert((int64_t)this->weight_sum_ + x.weight_sum_
           <= std::numeric_limits<int32_t>::max());
    this->weight_sum_ += x.weight_sum_;
    return *this;
  }

  AdjustedPrice GetAverage() const {
    AdjustedPrice result;
    result.SetRawValue((adjusted_price_sum_ + weight_sum_ / 2) / weight_sum_);
    return result;
  }

  int32_t GetWeight() const { return weight_sum_; }

 private:
  int64_t adjusted_price_sum_;
  int32_t weight_sum_;
};

class Features {
 public:
  Features() {}
  Features(const vector<Feature> features,
           const FeatureConfig& config,
           const string& name)
      : name_(name), config_(config), features_(features) {
    fprintf(stderr, "Features for %s\n", name_.c_str());
    InitTotalWeight();
    InitFuturePrice();
  }

  void Init(const vector<Feature> features,
            const FeatureConfig& config,
            const string& name) {
    name_ = name;
    config_ = config;
    features_ = features;
    fprintf(stderr, "Features for %s\n", name_.c_str());
    InitTotalWeight();
    InitFuturePrice();
  }

  void Init(const FeatureConfig& config,
            const AccumulatedRates& rates,
            int ratio_from,
            int ratio_to) {
    name_ = rates.GetName();
    config_ = config;
    fprintf(stderr, "Generating features for %s...\n", name_.c_str());

    Time next_time = rates.GetStartTime();
    mutex time_mutex;
    mutex features_mutex;
    int last_percentage = 0;
    fprintf(stderr, "- Processing 0%%...\n");

    TimeDifference time_step =
        TimeDifference::InMinute(max(1, ratio_from / 2));

    Parallel([&](int thread_id){
      Time time;
      {
        lock_guard<mutex> time_mutex_guard(time_mutex);
        time = next_time;
        if (!(time < rates.GetEndTime())) {
          return false;
        }
        next_time += time_step;
      }

      vector<Feature> features;
      for (int ratio = ratio_from; ratio <= ratio_to; ratio++) {
        Feature feature;
        if (feature.Init(config, rates, time, ratio)) {
          features.push_back(feature);
        }
      }

      {
        lock_guard<mutex> features_mutex_guard(features_mutex);
        for (const Feature& feature: features) {
          features_.push_back(feature);
        }
      }

      if (thread_id == 0) {
        int current_percentage =
            static_cast<int>(
                floor((time - rates.GetStartTime()).GetMinute()
                      / (rates.GetEndTime()
                         - rates.GetStartTime()).GetMinute() * 100));
        if (current_percentage != last_percentage) {
          fprintf(stderr,
                  CLEAR_LINE "- Processing %d%%...\n",
                  current_percentage);
          last_percentage = current_percentage;
        }
      }

      return true;
    });
    fprintf(stderr, CLEAR_LINE "Successfully processed.\n");

    fprintf(stderr, "- # of features for %s is %lu.\n",
            rates.GetName().c_str(), features_.size());

    InitTotalWeight();
    InitFuturePrice();
  }

  void InitTotalWeight() {
    total_weight_ = 0;
    for (const Feature& feature : features_) {
      total_weight_ += feature.GetWeight();
    }
    fprintf(stderr, "- Total weight: %d\n", total_weight_);
  }

  void InitFuturePrice() {
    for (const Feature& feature : features_) {
      future_stat_.Add(feature.GetFuturePrice());
    }
    fprintf(stderr, "- Future price stat: %s [%+.4f, %+.4f]\n",
            future_stat_.DebugString().c_str(),
            future_stat_.GetLowerBound().GetRatio(),
            future_stat_.GetUpperBound().GetRatio());
  }

  size_t size() const { return features_.size(); }
  vector<Feature>::const_iterator begin() const { return features_.begin(); }
  vector<Feature>::const_iterator end() const { return features_.end(); }

  const FeatureConfig& GetFeatureConfig() const { return config_; }
  const AdjustedPriceStat& GetFutureStat() const { return future_stat_; }

  void Cluster(int size, Features* output) const {
    fprintf(stderr, "Clustering features for %s...\n", name_.c_str());

    if ((int)features_.size() / 2 < size) {
      fprintf(stderr, "Not enough features: %d vs %lu", size, features_.size());
      exit(1);
    }

    vector<Feature> candidate_features;
    for (set<size_t> index_set; (int)index_set.size() < size;) {
      size_t new_index = Rand() % features_.size();
      if (index_set.count(new_index) > 0) { continue; }
      index_set.insert(new_index);
      candidate_features.push_back(features_[new_index]);
    }

    vector<FeatureSum> new_features(candidate_features.size());
    vector<int> new_features_count(candidate_features.size());
    int last_percentage = 0;
    fprintf(stderr, "Processing 0%%...\n");
    size_t next_feature_index = 0;
    mutex feature_index_mutex;
    mutex new_features_mutex;

    Parallel([&](int thread_id){
      size_t feature_index;
      {
        lock_guard<mutex> feature_index_mutex_guard(feature_index_mutex);
        feature_index = next_feature_index;
        if (!(feature_index < features_.size())) {
          return false;
        }
        next_feature_index++;
      }

      if (thread_id == 0) {
        int current_percentage =
            static_cast<int>(feature_index * 100 / features_.size());
        if (current_percentage != last_percentage) {
          fprintf(stderr, CLEAR_LINE "Processing %d%%...\n",
                  current_percentage);
          last_percentage = current_percentage;
        }
      }

      const Feature& feature = features_[feature_index];
      double min_distance = INFINITY;
      size_t min_index = 0;
      for (size_t index = 0; index < candidate_features.size(); index++) {
        const Feature& candidate_feature = candidate_features[index];
        double distance = feature.MeasureDistance(config_, candidate_feature);
        if (distance < min_distance) {
          min_distance = distance;
          min_index = index;
        }
      }
      assert(!IsInf(min_distance));

      {
        lock_guard<mutex> new_features_mutex_guard(new_features_mutex);
        new_features[min_index] += feature;
        new_features_count[min_index]++;
      }

      return true;
    });
    fprintf(stderr, CLEAR_LINE "Successfully processed.\n");

    vector<Feature> average_features;
    AdjustedPriceWeightedSum future_price;
    Features result;
    for (size_t i = 0; i < new_features.size(); i++) {
      assert(new_features_count[i] >= 0);
      assert(new_features_count[i] > 0);
      if (new_features_count[i] == 0) { continue; }
      Feature new_feature = new_features[i].GetAverage(new_features_count[i]);
      average_features.push_back(new_feature);
      future_price += AdjustedPriceWeightedSum(
          new_feature.GetFuturePrice(), new_feature.GetWeight());
    }
    fprintf(stderr, "Average future price: %.6f\n",
            future_price.GetAverage().GetRegularizedPrice().GetRealPrice());

    output->Init(average_features, config_, "clustered " + name_);
  }

  AdjustedPrice Predict(const Feature& target) const {
    assert(total_weight_ > 0);

    vector<pair<double, size_t>> distance_to_index;
    for (size_t feature_index = 0;
         feature_index < features_.size(); feature_index++) {
      const Feature& feature = features_[feature_index];
      distance_to_index.emplace_back(
          feature.MeasureDistance(config_, target), feature_index);
    }
    sort(distance_to_index.begin(), distance_to_index.end());
    AdjustedPriceWeightedSum result;
    for (const pair<double, size_t>& distance_and_index : distance_to_index) {
      const Feature& feature = features_[distance_and_index.second];
      const AdjustedPrice& price = feature.GetFuturePrice();
      assert(price.IsValid());
      result += AdjustedPriceWeightedSum(price, feature.GetWeight());
      if (result.GetWeight() > total_weight_ * kPredictRatio) { break; }
    }
    return result.GetAverage();
  }

  void Print(FILE* fp = nullptr, const string& indent = "") const {
    if (fp == nullptr) {
      fp = stdout;
    }

    fprintf(fp, "{\n%s  name: \"%s\"\n", indent.c_str(), name_.c_str());
    fprintf(fp, "%s  features: [\n", indent.c_str());
    for (const Feature& feature: features_) {
      fprintf(fp, "%s    %s\n",
              indent.c_str(), feature.DebugString(indent + "    ").c_str());
    }
    fprintf(fp, "%s  ],\n%s}", indent.c_str(), indent.c_str());
  }

 private:
  string name_;
  FeatureConfig config_;
  vector<Feature> features_;
  AdjustedPriceStat future_stat_;
  int32_t total_weight_;

  DISALLOW_COPY_AND_ASSIGN(Features);
};

struct Leverage {
 public:
  Leverage() : minimal_leverage_(NAN), maximal_leverage_(NAN) {}

  void Extend(double leverage) {
    if (IsNan(leverage)) { return; }

    if (IsNan(minimal_leverage_) || leverage < minimal_leverage_) {
      minimal_leverage_ = leverage;
    }
    if (IsNan(maximal_leverage_) || leverage > maximal_leverage_) {
      maximal_leverage_ = leverage;
    }
  }

  double GetMinimalLeverage() const { return minimal_leverage_; }
  double GetMaximalLeverage() const { return maximal_leverage_; }

 private:
  double minimal_leverage_;
  double maximal_leverage_;
};

struct Correlation {
 public:
  Correlation()
      : sum_(0), x_sum_(0), xx_sum_(0), y_sum_(0), yy_sum_(0), xy_sum_(0) {}

  void Add(double x, double y) {
    CHECK(!IsNan(x) && !IsInf(x)) << "X is invalid: " << x;
    CHECK(!IsNan(y) && !IsInf(y)) << "Y is invalid: " << y;
    sum_ += 1;
    x_sum_ += x;
    xx_sum_ += x * x;
    y_sum_ += y;
    yy_sum_ += y * y;
    xy_sum_ += x * y;
  }

  double GetWeight() const { return sum_; }

  double Calculate() const {
    CHECK_NE(sum_, 0) << "No data is available.";
    return (xy_sum_ * sum_ - x_sum_ * y_sum_)
             / sqrt(xx_sum_ * sum_ - x_sum_ * x_sum_)
             / sqrt(yy_sum_ * sum_ - y_sum_ * y_sum_);
  }

  string DebugString() const {
    return StringPrintf(
        "n: %.4f, E(x): %.4f, E(xx): %.4f, "
        "E(y): %.4f, E(yy): %.4f, E(xy): %.4f",
        sum_, x_sum_ / sum_, xx_sum_ / sum_,
        y_sum_ / sum_, yy_sum_ / sum_, xy_sum_ / sum_);
  }

 private:
  double sum_;
  double x_sum_;
  double xx_sum_;
  double y_sum_;
  double yy_sum_;
  double xy_sum_;
};

struct Correlations {
 public:
  Correlations() : sum_(0), x_sum_(0), xx_sum_(0) {}

  void Add(double x, double y) {
    correlation_.Add(x, y);
  }

  double Commit() {
    double weight = correlation_.GetWeight();
    CHECK(!IsNan(weight) && !IsInf(weight));
    if (weight < 1e-6) {
      return NAN;
    }

    double x = correlation_.Calculate();
    CHECK(!IsNan(x) && !IsInf(x))
        << "Bad correlation: " << x << ", " << correlation_.DebugString();
    sum_ += weight;
    x_sum_ += x * weight;
    xx_sum_ += x * x * weight;
    correlation_ = Correlation();
    return x;
  }

  string DebugString() const {
    return StringPrintf("%.4f±%.4f",
                        x_sum_ / sum_, 
                        sqrt(xx_sum_ / sum_ - x_sum_ * x_sum_ / sum_ / sum_));
  }

 private:
  Correlation correlation_;
  double sum_;
  double x_sum_;
  double xx_sum_;
};

class Simulator {
 public:
  Simulator(const AccumulatedRates* training_rates,
            const AccumulatedRates* test_rates)
      : training_rates_(training_rates),
        test_rates_(test_rates) {}

  void LearnQFeatures() {
    const int kRatio = 16;
    // const FeatureConfig feature_config({1, 6, 18}, 6);
    const FeatureConfig feature_config({1, 8, 32}, 8);
    // const FeatureConfig feature_config({1, 8, 32}, kRatio * 2);
    // const FeatureConfig feature_config({1, 6, 12}, 6);
    // const FeatureConfig feature_config({1, 4, 8}, 4);
    QFeatures features;
    // features.Init(feature_config, *training_rates_, 5, 3);
    features.Init(feature_config, *training_rates_, kRatio, kRatio / 2);
    // features.Init(feature_config, *training_rates_, 20, 12);
    // features.Init(feature_config, *training_rates_, 48, 24);
    for (double learning_rate = 1; ; learning_rate *= 0.999) {
      for (int i = 0; i < 10; i++) {
        // features.Replay();
        features.Learn(learning_rate * 0.9 + 0.1, kRatio);
      }
      features.Simulate(*test_rates_, kRatio);
      // features.Simulate(*test_rates_, 16);
    }
  }

  void EvaluateFeatures() {
    const FeatureConfig feature_config({1, 8, 32}, 8);
    // const FeatureConfig feature_config({1, 6, 12}, 6);
    // const FeatureConfig feature_config({1, 12, 24}, 12);
    // const FeatureConfig feature_config({1, 24, 48}, 24);

    Features clustered_features;
    {
      Features training_features;
      // training_features.Init(feature_config, *training_rates_, 4, 64);
      // training_features.Init(feature_config, *training_rates_, 32, 256);
      training_features.Init(feature_config, *training_rates_, 4, 64);
      // training_features.Init(feature_config, *training_rates_, 16, 64);
      training_features.Cluster(1000, &clustered_features);
    }

    const int kRatioSize = 7;
    Correlations correlations[kRatioSize];
    for (Time next_now = test_rates_->GetStartTime();
         next_now < test_rates_->GetEndTime();) {
      printf("%s:\n", next_now.DebugString().c_str());
      int32_t week_index = next_now.GetWeekIndex();

      mutex now_mutex;
      mutex correlation_mutex;
      Parallel([&](int){
        Time now;
        {
          lock_guard<mutex> now_mutex_guard(now_mutex);
          now = next_now;
          if (!(now < test_rates_->GetEndTime()) ||
              now.GetWeekIndex() != week_index) {
            return false;
          }
          next_now.AddMinute(1);
        }

        for (int ratio_index = 0; ratio_index < kRatioSize; ratio_index++) {
          const int ratio = 1 << ratio_index;
          Feature feature;
          if (!feature.Init(feature_config, *test_rates_, now, ratio) ||
              !feature.GetFuturePrice().IsValid()) {
            continue;
          }

          AdjustedPrice actual_price = feature.GetFuturePrice();
          feature.ClearFuturePrice();
          AdjustedPrice predicted_price = clustered_features.Predict(feature);

          {
            double x = log(actual_price.GetRatio());
            double y = log(predicted_price.GetRatio());
            lock_guard<mutex> correlation_mutex_guard(correlation_mutex);
            correlations[ratio_index].Add(x, y);
          }
        }

        return true;
      });
      for (int ratio_index = 0; ratio_index < kRatioSize; ratio_index++) {
        printf("Ratio: % 3d, Correlation: %.4f\n",
               1 << ratio_index,
               correlations[ratio_index].Commit());
      }
    }

    printf("Final commit:\n");
    for (int ratio_index = 0; ratio_index < kRatioSize; ratio_index++) {
      printf("Ratio: % 3d, Correlation: %s\n",
             1 << ratio_index,
             correlations[ratio_index].DebugString().c_str());
    }
  }

  AdjustedPrice Predict(const Features& model,
                        const FeatureConfig& feature_config,
                        Time now,
                        int ratio,
                        PriceDifference adjustment) const {
    Feature current_feature;
    if (!current_feature.Init(
            feature_config, *test_rates_, now, ratio, adjustment)) {
      return AdjustedPrice();
    }
    AdjustedPrice predicted_price = model.Predict(current_feature);
    return predicted_price;
  }

  void GenerateModel(
      const FeatureConfig& config,
      int ratio_from,
      int ratio_to,
      Features* output) {
    Features training_features;
    training_features.Init(config, *training_rates_, ratio_from, ratio_to);
    training_features.Cluster(1000, output);
  }

  bool ShouldExitPosition(const Features& model,
                          const FeatureConfig& config,
                          int position,
                          Time now) const {
    for (double ratio = 8.0; ratio < 16 + 1e-6; ratio *= sqrt(2.0)) {
      AdjustedPrice predicted_price = Predict(
          model, config, now, static_cast<int>(round(ratio)),
          PriceDifference::InRatio(1));
      if (!predicted_price.IsValid()) { return true; }
      if (position < 0 &&
          predicted_price < model.GetFutureStat().GetLowerBound()) {
        return false;
      }
      if (position > 0 &&
          predicted_price > model.GetFutureStat().GetUpperBound()) {
        return false;
      }
    }
    return true;
  }

  int GetPositionBias(const Features& model,
                      const FeatureConfig& config,
                      Time now,
                      PriceDifference adjustment) const {
    int position = 0;
    for (double ratio = 8.0; ratio < 64 + 1e-6; ratio *= sqrt(2.0)) {
      AdjustedPrice predicted_price = Predict(
          model, config, now, static_cast<int>(round(ratio)),
          adjustment);
      if (!predicted_price.IsValid()) {
        return 0;
      }
      int future_position = 0;
      if (predicted_price < model.GetFutureStat().GetLowerBound()) {
        future_position = -1;
      } else if (predicted_price > model.GetFutureStat().GetUpperBound()) {
        future_position = 1;
      } else {
        return 0;
      }
      if (position == 0) {
        position = future_position;
      } else if (position != future_position) {
        return 0;
      }
    }
    return position;
  }

  void SimulateDiscrete() {
//    FeatureConfig feature_config({1, 8, 64}, 8);
    FeatureConfig feature_config({1, 8, 64}, 8);
    Features model;
    GenerateModel(feature_config, 4, 32, &model);

    Asset base_asset;
    Asset asset;
    for (Time now = test_rates_->GetStartTime();
         now < test_rates_->GetEndTime(); now.AddMinute(1)) {
      Price current_price = test_rates_->GetRate(now, now).GetClosePrice();
      if (!current_price.IsValid()) { continue; }

      base_asset.Trade(now, current_price, 0.5);

      // 次の時刻の価格が存在しなければ強制決済
      if (!test_rates_->GetRate(
              now + TimeDifference::InMinute(1),
              now + TimeDifference::InMinute(1)).GetClosePrice().IsValid()) {
        asset.Trade(now, current_price, 0, true /* use_spread */);
        printf("%s: %s:\n",
               now.DebugString().c_str(),
               test_rates_->GetRate(now, now)
                  .GetClosePrice().DebugString().c_str());
        printf("  Settlement.\n");
        continue;
      }

      double current_leverage = asset.GetLeverage(current_price);
      int current_position =
          (fabs(current_leverage) < 1e-6) ? 0 : Sign(current_leverage);
      if (current_position != 0 &&
          !ShouldExitPosition(model, feature_config, current_position, now)) {
        continue;
      }

      bool initialized = false;
      int target_position = 0;
      for (int adjustment_index = -5; adjustment_index <= 5;
           adjustment_index++) {
        PriceDifference adjustment =
            PriceDifference::InRatio(1.00004) * adjustment_index;
        int position = GetPositionBias(
            model, feature_config, now, adjustment);
        if (!initialized) {
          target_position = position;
          initialized = true;
          continue;
        }
        if (target_position != position) {
          target_position = 0;
          break;
        }
      }

      if (current_position == target_position) {
        continue;
      }

      asset.Trade(now, current_price, target_position, true /* use_spread */);

      printf("%s: %s:\n",
             now.DebugString().c_str(),
             test_rates_->GetRate(now, now)
                .GetClosePrice().DebugString().c_str());
      printf("\t%.6f/%.6f (%.3f => %.3f) Fee: %.3f\n",
             asset.GetValue(current_price),
             base_asset.GetValue(current_price),
             current_leverage,
             asset.GetLeverage(current_price),
             asset.GetTotalFee());
    }
  }

  struct TradeState {
    Asset base_asset;
    Asset asset;
    string output;
  };

  TradeState ProcessTick(
      Time now, const Features& model, future<TradeState> current_state) {
    Price current_price = test_rates_->GetRate(now, now).GetClosePrice();
    if (!current_price.IsValid()) {
      return current_state.get();
    }

    // 次の時刻の価格が存在しなければ強制決済
    if (!test_rates_->GetRate(
            now + TimeDifference::InMinute(1),
            now + TimeDifference::InMinute(1)).GetClosePrice().IsValid()) {
      TradeState state = current_state.get();
      state.base_asset.Trade(now, current_price, 0.5);
      state.asset.Trade(now, current_price, 0, true /* use_spread */);
      return state;
    }

    double minimal_leverage = NAN;
    double maximal_leverage = NAN;
    double target_leverage = NAN;
    for (int adjustment = -5; adjustment <= 5; adjustment++) {
      double leverage_sum = 0.0;
      int failure = 0, success = 0;
      for (double ratio = 8; ratio < 32.001; ratio *= sqrt(2)) {
        Feature current_feature;
        if (!current_feature.Init(
                model.GetFeatureConfig(),
                *test_rates_, now, static_cast<int>(round(ratio)),
                PriceDifference::InRatio(1.00004) * adjustment)) {
          failure++;
          continue;
        }
        success++;

        AdjustedPrice predicted_aprice = model.Predict(current_feature);
        leverage_sum += -log(predicted_aprice.GetRatio());
      }
      double leverage = leverage_sum * 5000 / (failure + success);
      if (failure > success) {
        leverage = 0;
        // leverage_ratio = 1.0;
      }
      if (IsNan(minimal_leverage) || leverage < minimal_leverage) {
        minimal_leverage = leverage;
      }
      if (IsNan(maximal_leverage) || leverage > maximal_leverage) {
        maximal_leverage = leverage;
      }
      if (adjustment == 0) {
        target_leverage = leverage;
      }
    }

    TradeState state = current_state.get();
    double current_leverage = state.asset.GetLeverage(current_price);
    state.base_asset.Trade(now, current_price, 0.5);
    double leverage = target_leverage;
    leverage = max(-1.0, min(1.0, leverage));
    state.asset.Trade(now, current_price, leverage, true /* use_spread */);
    state.output = StringPrintf(
        "%.5f[%.3f] (%.3f => %.3f)",
        state.asset.GetValue(current_price),
        state.asset.GetTotalFee(),
        current_leverage,
        state.asset.GetLeverage(current_price));
    return state;
  }

  void Simulate() {
    FeatureConfig feature_config({1, 8, 32}, 8);
    Features model;
    GenerateModel(feature_config, 4, 64, &model);

    Time next_now = test_rates_->GetStartTime();
    promise<TradeState> initial_state;
    future<TradeState> next_state = initial_state.get_future();
    initial_state.set_value(TradeState());

    mutex now_mutex;
    mutex correlation_mutex;
    Parallel([&](int){
      Time now;
      Price current_price;
      future<TradeState> current_state;
      promise<TradeState> promised_state;
      {
        lock_guard<mutex> now_mutex_guard(now_mutex);
        do {
          now = next_now;
          if (!(now < test_rates_->GetEndTime())) {
            return false;
          }
          next_now.AddMinute(1);
          current_price = test_rates_->GetRate(now, now).GetClosePrice();
        } while (!current_price.IsValid());
        current_state = std::move(next_state);
        next_state = promised_state.get_future();
      }

      TradeState state = ProcessTick(now, model, std::move(current_state));
      if (now.GetWeeklyIndex() % 60 == 0) {
        printf("%s: %s: %s\n",
               now.DebugString().c_str(),
               current_price.DebugString().c_str(),
               state.output.c_str());
      }
      state.output.clear();
      promised_state.set_value(state);

      return true;
    });
  }

 private:
  const AccumulatedRates* training_rates_;
  const AccumulatedRates* test_rates_;

  DISALLOW_COPY_AND_ASSIGN(Simulator);
};

void Test() {
  assert(IsNan(NAN));
  assert(!IsNan(INFINITY));
  assert(!IsNan(0.0));

  fprintf(stderr, "sizeof(Price): %lu\n", sizeof(Price));
  fprintf(stderr, "sizeof(PriceSum): %lu\n", sizeof(PriceSum));
  SegmentTree<int32_t, const int32_t&, max<int32_t>>::Test();
  AccumulatedRates::Test();
  Volatility::Test();
  RunAllTests();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  FLAGS_alsologtostderr =
      static_cast<bool>(Params::GetInteger("FX_ALSOLOGTOSTDERR"));

  if (argc < 2) {
    fprintf(stderr, "Usage: %s training evaluation\n", argv[0]);
    return 1;
  }

  Params::Init();
  LOG(INFO) << "Parameters: " << GetParams().ShortDebugString();
  fprintf(stderr, "\n");

#ifdef NDEBUG
  if (strcmp(argv[1], "TEST") == 0) { Test(); return 0; }
#else
  Test();
  if (strcmp(argv[1], "TEST") == 0) { return 0; }
#endif

  if (argc != 3) {
    fprintf(stderr, "Usage: %s training evaluation\n", argv[0]);
    return 1;
  }

  unique_ptr<Rates> training_rates(new Rates());
  training_rates->Load(Split(argv[1], ','), true /* is_training */);
  unique_ptr<AccumulatedRates> training_accumulated_rates(
      new AccumulatedRates("training data", *training_rates));
  training_accumulated_rates->InitDailyVolatility(*training_rates);

  unique_ptr<Rates> test_rates(new Rates());
  test_rates->Load(Split(argv[2], ','), false /* is_training */);
  unique_ptr<AccumulatedRates> test_accumulated_rates(
      new AccumulatedRates("test data", *test_rates));
  test_accumulated_rates->InitDailyVolatility(*training_rates);

  training_rates.reset();
  test_rates.reset();

  Simulator simulator(training_accumulated_rates.get(),
                      test_accumulated_rates.get());
  switch (GetParams().mode) {
    case Params::Mode::SIMULATE: {
      simulator.Simulate();
      break;
    }
    case Params::Mode::EVALUATE: {
      simulator.EvaluateFeatures();
      break;
    }
    case Params::Mode::QLEARN: {
      simulator.LearnQFeatures();
    }
  }

  return 0;
}
