import { useEffect, useMemo, useState } from 'react'

type FeatureKey = 'sepalLength' | 'sepalWidth' | 'petalLength' | 'petalWidth'

type FeatureState = Record<FeatureKey, number>

type PredictionResponse = {
  prediction_id: number
  prediction_label: string
  probabilities: Record<string, number>
}

type TrainResponse = {
  status: string
  accuracy: number
  train_time: number
  feature_importances: Record<string, number>
  message: string
}

const defaultFeatures: FeatureState = {
  sepalLength: 5.1,
  sepalWidth: 3.5,
  petalLength: 1.4,
  petalWidth: 0.2,
}

const featureMeta: Array<{ key: FeatureKey; label: string; unit: string }> = [
  { key: 'sepalLength', label: '花萼長度', unit: 'cm' },
  { key: 'sepalWidth', label: '花萼寬度', unit: 'cm' },
  { key: 'petalLength', label: '花瓣長度', unit: 'cm' },
  { key: 'petalWidth', label: '花瓣寬度', unit: 'cm' },
]

const getApiBaseUrl = () => {
  const envBase = import.meta.env.VITE_API_BASE_URL as string | undefined
  if (envBase) return envBase
  return 'https://2027-07-03.onrender.com'
}

const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`

function App() {
  const [activeTab, setActiveTab] = useState<'predict' | 'train'>('predict')
  const [features, setFeatures] = useState<FeatureState>(defaultFeatures)
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [predictionLoading, setPredictionLoading] = useState(false)
  const [predictionError, setPredictionError] = useState<string | null>(null)
  const [trainForm, setTrainForm] = useState({
    n_estimators: 100,
    max_depth: 10,
    test_size: 0.2,
    random_state: 42,
  })
  const [trainResult, setTrainResult] = useState<TrainResponse | null>(null)
  const [trainLoading, setTrainLoading] = useState(false)
  const [trainError, setTrainError] = useState<string | null>(null)

  useEffect(() => {
    void runPrediction(defaultFeatures)
  }, [])

  const handleFeatureChange = (key: FeatureKey, value: number) => {
    setFeatures((prev) => ({ ...prev, [key]: value }))
  }

  const handleTrainChange = (key: keyof typeof trainForm, value: number) => {
    setTrainForm((prev) => ({ ...prev, [key]: value }))
  }

  const runPrediction = async (nextFeatures: FeatureState) => {
    setPredictionLoading(true)
    setPredictionError(null)

    try {
      const response = await fetch(`${getApiBaseUrl()}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sepal_length: nextFeatures.sepalLength,
          sepal_width: nextFeatures.sepalWidth,
          petal_length: nextFeatures.petalLength,
          petal_width: nextFeatures.petalWidth,
        }),
      })

      if (!response.ok) {
        throw new Error('預測請求失敗，請確認後端服務是否可用。')
      }

      const data: PredictionResponse = await response.json()
      setPrediction(data)
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : '預測失敗')
      setPrediction(null)
    } finally {
      setPredictionLoading(false)
    }
  }

  const handlePredict = async (event: React.FormEvent) => {
    event.preventDefault()
    await runPrediction(features)
  }

  const handleTrain = async () => {
    setTrainLoading(true)
    setTrainError(null)

    try {
      const response = await fetch(`${getApiBaseUrl()}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n_estimators: trainForm.n_estimators,
          max_depth: trainForm.max_depth === 0 ? null : trainForm.max_depth,
          test_size: trainForm.test_size,
          random_state: trainForm.random_state,
        }),
      })

      if (!response.ok) {
        throw new Error('訓練請求失敗，請確認後端服務是否可用。')
      }

      const data: TrainResponse = await response.json()
      setTrainResult(data)
    } catch (error) {
      setTrainError(error instanceof Error ? error.message : '訓練失敗')
    } finally {
      setTrainLoading(false)
    }
  }

  const importanceEntries = useMemo(() => {
    if (!trainResult?.feature_importances) return []
    return Object.entries(trainResult.feature_importances).sort(([, a], [, b]) => b - a)
  }, [trainResult])

  const topPrediction = prediction?.probabilities
    ? Object.entries(prediction.probabilities).sort(([, a], [, b]) => b - a)[0]
    : null

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(52,211,153,0.18),_transparent_35%),linear-gradient(135deg,_#f8fafc_0%,_#eefcf7_45%,_#fdf2f8_100%)] px-4 py-6 text-slate-800 sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 rounded-[32px] border border-white/70 bg-white/70 p-4 shadow-[0_30px_80px_rgba(15,23,42,0.12)] backdrop-blur-xl sm:p-6 lg:p-8">
        <header className="overflow-hidden rounded-[28px] bg-slate-950 px-6 py-8 text-white sm:px-8 lg:px-10">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-2xl">
              <p className="mb-3 inline-flex rounded-full border border-emerald-400/30 bg-emerald-500/10 px-3 py-1 text-sm font-semibold text-emerald-200">
                🌸 Iris Flower ML Studio
              </p>
              <h1 className="text-2xl font-black tracking-tight sm:text-3xl lg:text-4xl">
                以現代前端打造的鳶尾花分類體驗
              </h1>
              <p className="mt-3 text-sm leading-6 text-slate-300 sm:text-base">
                此介面直接串接 FastAPI 後端，讓你能即時預測花卉品種，也能在線上重新訓練模型並查看評估結果與特徵重要性。
              </p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/10 px-4 py-4 text-sm text-slate-200">
              <p className="font-semibold text-white">後端服務</p>
              <p className="mt-1 text-slate-300">{getApiBaseUrl()}</p>
              <p className="mt-2 text-emerald-300">支援 /predict 與 /train API</p>
            </div>
          </div>
        </header>

        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => setActiveTab('predict')}
            className={`rounded-full px-5 py-2.5 text-sm font-semibold transition ${activeTab === 'predict' ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/20' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'}`}
          >
            🔮 即時預測
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('train')}
            className={`rounded-full px-5 py-2.5 text-sm font-semibold transition ${activeTab === 'train' ? 'bg-sky-500 text-white shadow-lg shadow-sky-500/20' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'}`}
          >
            ⚙️ 線上訓練
          </button>
        </div>

        {activeTab === 'predict' ? (
          <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
            <form onSubmit={handlePredict} className="rounded-[26px] border border-slate-200 bg-slate-50/80 p-5 shadow-sm sm:p-6">
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-slate-900">輸入花卉特徵</h2>
                  <p className="mt-1 text-sm text-slate-600">調整滑桿後點擊預測，立即得到模型結果。</p>
                </div>
                <button
                  type="submit"
                  className="rounded-full bg-emerald-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-600"
                >
                  {predictionLoading ? '預測中...' : '🔮 開始預測'}
                </button>
              </div>

              <div className="space-y-4">
                {featureMeta.map((item) => (
                  <label key={item.key} className="block rounded-2xl border border-slate-200 bg-white p-4">
                    <div className="mb-3 flex items-center justify-between text-sm font-semibold text-slate-700">
                      <span>{item.label}</span>
                      <span className="text-emerald-600">{features[item.key].toFixed(1)} {item.unit}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="10"
                      step="0.1"
                      value={features[item.key]}
                      onChange={(event) => handleFeatureChange(item.key, Number(event.target.value))}
                      className="w-full accent-emerald-500"
                    />
                  </label>
                ))}
              </div>
            </form>

            <div className="rounded-[26px] border border-slate-200 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 p-5 text-white shadow-sm sm:p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold">預測結果</h2>
                  <p className="mt-1 text-sm text-slate-300">依照目前輸入值，模型會回傳最可能的品種與機率分佈。</p>
                </div>
              </div>

              {predictionError ? (
                <div className="mt-6 rounded-2xl border border-rose-400/40 bg-rose-500/10 p-4 text-sm text-rose-200">
                  {predictionError}
                </div>
              ) : prediction ? (
                <div className="mt-6 space-y-4">
                  <div className="rounded-3xl border border-white/10 bg-white/10 p-5">
                    <p className="text-sm uppercase tracking-[0.3em] text-slate-300">預測品種</p>
                    <h3 className="mt-2 text-3xl font-black">{prediction.prediction_label}</h3>
                    <p className="mt-2 text-sm text-slate-300">
                      最高機率：{topPrediction ? formatPercent(topPrediction[1]) : '—'}
                    </p>
                  </div>

                  <div className="rounded-3xl border border-white/10 bg-white/10 p-5">
                    <h4 className="text-lg font-semibold">機率分佈</h4>
                    <div className="mt-4 space-y-3">
                      {Object.entries(prediction.probabilities).map(([name, value]) => (
                        <div key={name}>
                          <div className="mb-1 flex items-center justify-between text-sm text-slate-200">
                            <span>{name}</span>
                            <span>{formatPercent(value)}</span>
                          </div>
                          <div className="h-2.5 rounded-full bg-slate-700">
                            <div className="h-2.5 rounded-full bg-emerald-400" style={{ width: `${value * 100}%` }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="mt-6 rounded-3xl border border-dashed border-white/20 bg-white/5 p-6 text-sm text-slate-300">
                  目前尚未收到預測結果，請點擊按鈕開始分析。
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="grid gap-6 xl:grid-cols-[1fr_0.9fr]">
            <div className="rounded-[26px] border border-slate-200 bg-slate-50/80 p-5 shadow-sm sm:p-6">
              <div className="mb-6 flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-bold text-slate-900">調整訓練超參數</h2>
                  <p className="mt-1 text-sm text-slate-600">將模型重新訓練後，將即時更新評估指標與特徵重要性。</p>
                </div>
                <button
                  type="button"
                  onClick={handleTrain}
                  className="rounded-full bg-sky-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-sky-600"
                >
                  {trainLoading ? '訓練中...' : '🚀 開始訓練'}
                </button>
              </div>

              <div className="space-y-4">
                <label className="block rounded-2xl border border-slate-200 bg-white p-4">
                  <div className="mb-2 flex items-center justify-between text-sm font-semibold text-slate-700">
                    <span>決策樹數量</span>
                    <span className="text-sky-600">{trainForm.n_estimators}</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="500"
                    step="10"
                    value={trainForm.n_estimators}
                    onChange={(event) => handleTrainChange('n_estimators', Number(event.target.value))}
                    className="w-full accent-sky-500"
                  />
                </label>

                <label className="block rounded-2xl border border-slate-200 bg-white p-4">
                  <div className="mb-2 flex items-center justify-between text-sm font-semibold text-slate-700">
                    <span>最大樹深度</span>
                    <span className="text-sky-600">{trainForm.max_depth === 0 ? '無限制' : trainForm.max_depth}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="20"
                    step="1"
                    value={trainForm.max_depth}
                    onChange={(event) => handleTrainChange('max_depth', Number(event.target.value))}
                    className="w-full accent-sky-500"
                  />
                </label>

                <label className="block rounded-2xl border border-slate-200 bg-white p-4">
                  <div className="mb-2 flex items-center justify-between text-sm font-semibold text-slate-700">
                    <span>測試集比例</span>
                    <span className="text-sky-600">{trainForm.test_size.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.5"
                    step="0.05"
                    value={trainForm.test_size}
                    onChange={(event) => handleTrainChange('test_size', Number(event.target.value))}
                    className="w-full accent-sky-500"
                  />
                </label>

                <label className="block rounded-2xl border border-slate-200 bg-white p-4">
                  <div className="mb-2 flex items-center justify-between text-sm font-semibold text-slate-700">
                    <span>隨機種子</span>
                    <span className="text-sky-600">{trainForm.random_state}</span>
                  </div>
                  <input
                    type="number"
                    min="1"
                    value={trainForm.random_state}
                    onChange={(event) => handleTrainChange('random_state', Number(event.target.value))}
                    className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none focus:border-sky-400"
                  />
                </label>
              </div>
            </div>

            <div className="rounded-[26px] border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
              <h2 className="text-2xl font-bold text-slate-900">訓練結果</h2>
              <p className="mt-1 text-sm text-slate-600">模型訓練完成後，會把準確度與重要特徵呈現在這裡。</p>

              {trainError ? (
                <div className="mt-6 rounded-2xl border border-rose-400/30 bg-rose-50 p-4 text-sm text-rose-700">
                  {trainError}
                </div>
              ) : trainResult ? (
                <div className="mt-6 space-y-5">
                  <div className="grid gap-3 sm:grid-cols-3">
                    <div className="rounded-2xl bg-slate-50 p-4">
                      <p className="text-xs uppercase tracking-[0.25em] text-slate-500">準確度</p>
                      <p className="mt-2 text-2xl font-black text-emerald-600">{(trainResult.accuracy * 100).toFixed(2)}%</p>
                    </div>
                    <div className="rounded-2xl bg-slate-50 p-4">
                      <p className="text-xs uppercase tracking-[0.25em] text-slate-500">訓練耗時</p>
                      <p className="mt-2 text-2xl font-black text-sky-600">{trainResult.train_time.toFixed(3)}s</p>
                    </div>
                    <div className="rounded-2xl bg-slate-50 p-4">
                      <p className="text-xs uppercase tracking-[0.25em] text-slate-500">狀態</p>
                      <p className="mt-2 text-lg font-semibold text-slate-800">{trainResult.status}</p>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-slate-200 p-4">
                    <h3 className="text-lg font-semibold text-slate-900">特徵重要性</h3>
                    <div className="mt-4 space-y-3">
                      {importanceEntries.map(([name, value]) => (
                        <div key={name}>
                          <div className="mb-1 flex items-center justify-between text-sm text-slate-700">
                            <span>{name}</span>
                            <span>{(value * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2.5 rounded-full bg-slate-100">
                            <div className="h-2.5 rounded-full bg-violet-500" style={{ width: `${value * 100}%` }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-2xl bg-slate-50 p-4 text-sm text-slate-600">{trainResult.message}</div>
                </div>
              ) : (
                <div className="mt-6 rounded-3xl border border-dashed border-slate-200 bg-slate-50 p-6 text-sm text-slate-500">
                  尚未進行訓練，請點擊按鈕讓模型重新估計。
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
