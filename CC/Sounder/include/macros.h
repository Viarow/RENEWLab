#ifndef MACROS_H
#define MACROS_H

#include <atomic>
#include <vector>

#ifdef USE_UHD
static constexpr bool kUseUHD = true;
#else
static constexpr bool kUseUHD = false;
#endif

static constexpr size_t kStreamContinuous = 1;
static constexpr size_t kStreamEndBurst = 2;
static constexpr size_t kDsDimsNum = 5;
static constexpr size_t kDsDimSymbol = 2;

#define DEBUG_PRINT (0)
#define DEBUG_RADIO (0)
#define DEBUG_PLOT (0)

// TASK & SOCKET thread number
#define TASK_THREAD_NUM (1)
#define RX_THREAD_NUM (4)

#define MAX_FRAME_INC (2000)
#define TIME_DELTA (40) //ms
#define SETTLE_TIME_MS (1)
#define UHD_INIT_TIME_SEC (3) // radio init time for UHD devices
#define BEACON_INTERVAL (20) // frames

enum SchedulerEventType {
    kEventRxSymbol = 0,
    kTaskRecord = 1,
    kTaskRead = 2,
    kThreadTermination = 3
};

// each thread has a SampleBuffer
struct SampleBuffer {
    std::vector<char> buffer;
    std::atomic_int* pkg_buf_inuse;
};

struct Event_data {
    SchedulerEventType event_type;
    int data;
    int ant_id;
    size_t rx_buff_size;
    SampleBuffer* rx_buffer;
};

struct Package {
    uint32_t frame_id;
    uint32_t symbol_id;
    uint32_t cell_id;
    uint32_t ant_id;
    short data[];
    Package(int f, int s, int c, int a)
        : frame_id(f)
        , symbol_id(s)
        , cell_id(c)
        , ant_id(a)
    {
    }
};

#endif
