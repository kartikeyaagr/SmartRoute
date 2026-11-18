{{- define "smartroute.labels" -}}
app.kubernetes.io/name: smartroute
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}

{{- define "smartroute.selector" -}}
app.kubernetes.io/name: smartroute
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end }}
