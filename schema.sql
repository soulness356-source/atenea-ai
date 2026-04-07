-- Escuelas
create table if not exists escuelas (
  id uuid primary key default gen_random_uuid(),
  nombre text not null,
  ciudad text,
  created_at timestamptz default now()
);

-- Usuarios (maestros y directivos)
create table if not exists usuarios (
  id uuid primary key default gen_random_uuid(),
  nombre text not null,
  apellidos text not null,
  email text unique not null,
  password_hash text not null,
  rol text not null check (rol in ('maestro', 'directivo')),
  escuela_id uuid references escuelas(id),
  grupos_asignados text,
  created_at timestamptz default now()
);

-- Alumnos
create table if not exists alumnos (
  id uuid primary key default gen_random_uuid(),
  nombre text not null,
  apellidos text not null,
  matricula text not null,
  grupo text not null,
  semestre int not null,
  escuela_id uuid references escuelas(id),
  maestro_id uuid references usuarios(id),
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Registros de riesgo (histórico por alumno)
create table if not exists registros_riesgo (
  id uuid primary key default gen_random_uuid(),
  alumno_id uuid references alumnos(id) on delete cascade,
  fecha date default current_date,
  -- Variables académicas
  promedio_general numeric(4,2),
  asistencia_pct numeric(5,2),
  materias_reprobadas int default 0,
  tareas_entregadas_pct numeric(5,2),
  llegadas_tarde int default 0,
  reportes_disciplinarios int default 0,
  -- Variables psicoemocionales (1-4)
  motivacion int check (motivacion between 1 and 4),
  apoyo_familiar int check (apoyo_familiar between 1 and 4),
  nivel_estres int check (nivel_estres between 1 and 4),
  sentido_pertenencia int check (sentido_pertenencia between 1 and 4),
  expectativas_futuro int check (expectativas_futuro between 1 and 4),
  -- Resultado del modelo
  score_riesgo numeric(5,2),
  nivel_riesgo text check (nivel_riesgo in ('Alto', 'Medio', 'Bajo')),
  factores_principales text,
  created_at timestamptz default now()
);
